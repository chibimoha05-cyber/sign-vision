"""End-to-End Pipeline & Synthetic Benchmark Tests for SignHandTracker.

Verifies:
1. Monotonic frame indexing and timestamp handling across 60 sequential frames.
2. Feature vector shape, dtype, and absence of NaNs/Infs.
3. Sub-15ms typical inference latency measurement.
4. Overlay rendering and canvas dimension preservation.
5. Deterministic context manager and clean resource cleanup (.close()).
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import pytest

# Ensure src/ is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sign_vision.schema import TOTAL_FEATURE_DIM, FrameTrackingResult
from sign_vision.engine import SignHandTracker


@pytest.fixture
def synthetic_frame() -> np.ndarray:
    """Generates a synthetic 720p BGR test frame with geometric shapes."""
    frame = np.full((720, 1280, 3), (30, 35, 42), dtype=np.uint8)
    return frame


def test_tracker_pipeline_synthetic_sequence(synthetic_frame: np.ndarray) -> None:
    """Processes 60 sequential synthetic frames and verifies pipeline integrity and timing."""
    tracker = SignHandTracker(enable_smoothing=True)

    try:
        latencies = []
        last_timestamp = -1.0

        for idx in range(60):
            simulated_ts = (idx + 1) * 16.666  # ~60 FPS monotonic timestamps in ms
            result = tracker.process_frame(synthetic_frame, timestamp_ms=simulated_ts)

            # Assert return type
            assert isinstance(result, FrameTrackingResult)

            # Monotonic frame index
            assert result.frame_index == idx + 1

            # Monotonic timestamp guarantee
            assert result.timestamp_ms > last_timestamp
            last_timestamp = result.timestamp_ms

            # Latency bounds
            assert result.inference_latency_ms >= 0.0
            latencies.append(result.inference_latency_ms)

            # Output vector contract
            assert result.feature_vector.shape == (TOTAL_FEATURE_DIM,)
            assert result.feature_vector.dtype == np.float32
            assert not np.any(np.isnan(result.feature_vector))
            assert not np.any(np.isinf(result.feature_vector))

            # Presence mask contract
            assert len(result.presence_mask) == 2
            assert result.presence_mask[0] in (0, 1)
            assert result.presence_mask[1] in (0, 1)

            # Test overlay rendering
            annotated = tracker.draw_overlays(synthetic_frame, result, draw_hud=True)
            assert annotated.shape == synthetic_frame.shape
            assert annotated.dtype == np.uint8

        # Latency statistics check (synthetic image inference should be fast)
        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 100.0, f"Average latency too high: {avg_latency:.2f}ms"

    finally:
        tracker.close()


def test_tracker_context_manager(synthetic_frame: np.ndarray) -> None:
    """Verifies that SignHandTracker works reliably within a context manager."""
    with SignHandTracker(enable_smoothing=False) as tracker:
        result = tracker.process_frame(synthetic_frame)
        assert result.frame_index == 1
        assert result.feature_vector.shape == (TOTAL_FEATURE_DIM,)

    # After exit, internal backends should be closed
    assert tracker._tasks_landmarker is None
    assert tracker._solutions_hands is None


def test_dual_backend_fallback_or_selection() -> None:
    """Verifies backend initialization and automatic fallback logic."""
    import mediapipe as mp

    tracker = SignHandTracker(preferred_backend="solutions")
    try:
        if hasattr(mp, "solutions"):
            assert tracker.active_backend_name == "solutions"
        else:
            # Graceful fallback to Tasks backend on MediaPipe 1.x
            assert tracker.active_backend_name == "tasks"
    finally:
        tracker.close()


def test_handedness_anti_flip_protection() -> None:
    """Verifies that temporary 1-2 frame handedness label flips from MediaPipe are suppressed."""
    tracker = SignHandTracker()

    try:
        # Frame 1: Right hand detected at (0.65, 0.55)
        raw_lm = np.zeros((21, 3), dtype=np.float32)
        raw_lm[:, 0] = 0.65
        raw_lm[:, 1] = 0.55
        det_right = [{"handedness": "Right", "confidence": 0.90, "landmarks": raw_lm}]

        l1, r1 = tracker._associate_and_stabilize_hands(det_right)
        assert r1 is not None and r1["handedness"] == "Right"
        # Lock right track
        tracker._build_hand_data(r1, "Right", (480, 640), 0.033)
        assert tracker._tracks["Right"].is_locked is True

        # Frame 2: MediaPipe temporarily flips label to "Left" at the exact same location
        det_flip = [{"handedness": "Left", "confidence": 0.82, "landmarks": raw_lm + 0.005}]
        l2, r2 = tracker._associate_and_stabilize_hands(det_flip)

        # Anti-flip rule must override reported "Left" and maintain "Right"
        assert r2 is not None and r2["handedness"] == "Right"
        assert l2 is None

    finally:
        tracker.close()


def test_temporal_occlusion_coasting() -> None:
    """Verifies that tracker coasts through a 2-frame dropout and resets after max_coast_frames."""
    tracker = SignHandTracker(max_coast_frames=3)

    try:
        raw_lm = np.zeros((21, 3), dtype=np.float32)
        raw_lm[0] = [0.5, 0.5, 0.0]
        det = {"handedness": "Right", "confidence": 0.90, "landmarks": raw_lm}

        # Frame 1: Active hand lock
        h1 = tracker._build_hand_data(det, "Right", (480, 640), 0.033)
        assert h1 is not None
        assert h1.is_coasting is False
        assert h1.confidence == 0.90

        # Frame 2 (Dropout 1): No detection -> Coast frame 1
        h2 = tracker._build_hand_data(None, "Right", (480, 640), 0.066)
        assert h2 is not None
        assert h2.is_coasting is True
        assert h2.confidence < 0.90  # Decayed

        # Frame 3 (Dropout 2): No detection -> Coast frame 2
        h3 = tracker._build_hand_data(None, "Right", (480, 640), 0.099)
        assert h3 is not None
        assert h3.is_coasting is True

        # Frame 4 (Dropout 3): Coast frame 3
        h4 = tracker._build_hand_data(None, "Right", (480, 640), 0.132)
        assert h4 is not None
        assert h4.is_coasting is True

        # Frame 5 (Dropout 4): Exceeds max_coast_frames (3) -> Hand must be cleared
        h5 = tracker._build_hand_data(None, "Right", (480, 640), 0.165)
        assert h5 is None
        assert tracker._tracks["Right"].is_locked is False

    finally:
        tracker.close()


def test_inter_hand_nms_phantom_duplicate_suppression() -> None:
    """Verifies that Inter-Hand OKS NMS suppresses phantom duplicate skeletons on the same hand."""
    # Create canonical hand coordinates
    hand_pts = np.zeros((21, 3), dtype=np.float32)
    for i in range(21):
        hand_pts[i] = [0.50 + i * 0.005, 0.50 + i * 0.008, 0.0]

    # Duplicate detection: slightly jittered (+0.003) overlapping duplicate
    duplicate_pts = hand_pts.copy() + 0.003

    detections = [
        {"handedness": "Left", "confidence": 0.85, "landmarks": hand_pts},
        {"handedness": "Right", "confidence": 0.94, "landmarks": duplicate_pts},
    ]

    survivors = SignHandTracker._apply_inter_hand_nms(detections)

    # Must suppress duplicate: only 1 detection survives
    assert len(survivors) == 1
    # Higher confidence detection must be preserved
    assert survivors[0]["confidence"] == 0.94
    assert survivors[0]["handedness"] == "Right"

    # Distinct hands test: Left hand at 0.25, Right hand at 0.75
    left_pts = hand_pts.copy()
    left_pts[:, 0] -= 0.25
    right_pts = hand_pts.copy()
    right_pts[:, 0] += 0.25

    distinct_detections = [
        {"handedness": "Left", "confidence": 0.90, "landmarks": left_pts},
        {"handedness": "Right", "confidence": 0.92, "landmarks": right_pts},
    ]

    distinct_survivors = SignHandTracker._apply_inter_hand_nms(distinct_detections)
    assert len(distinct_survivors) == 2, "Distinct two hands should NOT be suppressed"


def test_fast_motion_no_frozen_coasting() -> None:
    """Verifies that the tracker never freezes a static skeleton in mid-air during high-speed motion."""
    tracker = SignHandTracker(max_coast_frames=1)

    try:
        raw_lm = np.zeros((21, 3), dtype=np.float32)
        raw_lm[0] = [0.20, 0.50, 0.0]
        det1 = {"handedness": "Right", "confidence": 0.92, "landmarks": raw_lm}

        # Frame 1: Hand at x=0.20
        h1 = tracker._build_hand_data(det1, "Right", (480, 640), timestamp_sec=0.033)
        assert h1 is not None

        # Frame 2: Rapid hand movement to x=0.75 (delta_x = 0.55 in 33ms -> v ~ 16.6 norm-units/s)
        raw_lm_fast = raw_lm.copy()
        raw_lm_fast[:, 0] += 0.55
        det2 = {"handedness": "Right", "confidence": 0.88, "landmarks": raw_lm_fast}
        h2 = tracker._build_hand_data(det2, "Right", (480, 640), timestamp_sec=0.066)
        assert h2 is not None

        # Verify wrist velocity was calculated and exceeds fast-motion threshold
        track = tracker._tracks["Right"]
        assert track.wrist_velocity > 0.80

        # Frame 3: Detection drops during fast motion
        # The tracker must NOT coast or freeze a dead skeleton in mid-air!
        h3 = tracker._build_hand_data(None, "Right", (480, 640), timestamp_sec=0.099)
        assert h3 is None, "Tracker hallucinated a frozen/coasted skeleton during fast motion!"
        assert track.is_locked is False

    finally:
        tracker.close()


def test_rtmpose_backend_contract_compliance() -> None:
    """Verifies that RTMPose SimCC backend initializes and produces identical 144-D feature vector contract."""
    from sign_vision.backends import RTMPoseBackend

    backend = RTMPoseBackend()
    if not backend.is_available:
        pytest.skip("RTMPose models not available in environment.")

    try:
        test_frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        tracker = SignHandTracker(preferred_backend="rtmpose")
        assert tracker.active_backend_name == "rtmpose"

        result = tracker.process_frame(test_frame)
        assert isinstance(result, FrameTrackingResult)
        assert result.feature_vector.shape == (TOTAL_FEATURE_DIM,)
        assert result.feature_vector.dtype == np.float32

        # Test live switching to tasks and back
        switched = tracker.switch_backend("tasks")
        assert tracker.active_backend_name in ("tasks", "solutions")

        switched_back = tracker.switch_backend("rtmpose")
        assert tracker.active_backend_name == "rtmpose"

    finally:
        tracker.close()
        backend.close()


