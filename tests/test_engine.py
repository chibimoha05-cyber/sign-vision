"""End-to-end pipeline and synthetic benchmark tests for SignHandTracker."""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import pytest

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
            simulated_ts = (idx + 1) * 16.666
            result = tracker.process_frame(synthetic_frame, timestamp_ms=simulated_ts)

            assert isinstance(result, FrameTrackingResult)

            assert result.frame_index == idx + 1

            assert result.timestamp_ms > last_timestamp
            last_timestamp = result.timestamp_ms

            assert result.inference_latency_ms >= 0.0
            latencies.append(result.inference_latency_ms)

            assert result.feature_vector.shape == (TOTAL_FEATURE_DIM,)
            assert result.feature_vector.dtype == np.float32
            assert not np.any(np.isnan(result.feature_vector))
            assert not np.any(np.isinf(result.feature_vector))

            assert len(result.presence_mask) == 2
            assert result.presence_mask[0] in (0, 1)
            assert result.presence_mask[1] in (0, 1)

            annotated = tracker.draw_overlays(synthetic_frame, result, draw_hud=True)
            assert annotated.shape == synthetic_frame.shape
            assert annotated.dtype == np.uint8

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
            assert tracker.active_backend_name == "tasks"
    finally:
        tracker.close()


def test_handedness_anti_flip_protection() -> None:
    """Verifies that temporary 1-2 frame handedness label flips from MediaPipe are suppressed."""
    tracker = SignHandTracker()

    try:
        raw_lm = np.zeros((21, 3), dtype=np.float32)
        raw_lm[:, 0] = 0.65
        raw_lm[:, 1] = 0.55
        det_right = [{"handedness": "Right", "confidence": 0.90, "landmarks": raw_lm}]

        l1, r1 = tracker._associate_and_stabilize_hands(det_right)
        assert r1 is not None and r1["handedness"] == "Right"
        tracker._build_hand_data(r1, "Right", (480, 640), 0.033)
        assert tracker._tracks["Right"].is_locked is True

        det_flip = [{"handedness": "Left", "confidence": 0.82, "landmarks": raw_lm + 0.005}]
        l2, r2 = tracker._associate_and_stabilize_hands(det_flip)

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

        h1 = tracker._build_hand_data(det, "Right", (480, 640), 0.033)
        assert h1 is not None
        assert h1.is_coasting is False
        assert h1.confidence == 0.90

        h2 = tracker._build_hand_data(None, "Right", (480, 640), 0.066)
        assert h2 is not None
        assert h2.is_coasting is True
        assert h2.confidence < 0.90

        h3 = tracker._build_hand_data(None, "Right", (480, 640), 0.099)
        assert h3 is not None
        assert h3.is_coasting is True

        h4 = tracker._build_hand_data(None, "Right", (480, 640), 0.132)
        assert h4 is not None
        assert h4.is_coasting is True

        h5 = tracker._build_hand_data(None, "Right", (480, 640), 0.165)
        assert h5 is None
        assert tracker._tracks["Right"].is_locked is False

    finally:
        tracker.close()


def test_inter_hand_nms_phantom_duplicate_suppression() -> None:
    """Verifies that Inter-Hand OKS NMS suppresses phantom duplicate skeletons on the same hand."""
    hand_pts = np.zeros((21, 3), dtype=np.float32)
    for i in range(21):
        hand_pts[i] = [0.50 + i * 0.005, 0.50 + i * 0.008, 0.0]

    duplicate_pts = hand_pts.copy() + 0.003

    detections = [
        {"handedness": "Left", "confidence": 0.85, "landmarks": hand_pts},
        {"handedness": "Right", "confidence": 0.94, "landmarks": duplicate_pts},
    ]

    survivors = SignHandTracker._apply_inter_hand_nms(detections)

    assert len(survivors) == 1
    assert survivors[0]["confidence"] == 0.94
    assert survivors[0]["handedness"] == "Right"

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

        h1 = tracker._build_hand_data(det1, "Right", (480, 640), timestamp_sec=0.033)
        assert h1 is not None

        raw_lm_fast = raw_lm.copy()
        raw_lm_fast[:, 0] += 0.55
        det2 = {"handedness": "Right", "confidence": 0.88, "landmarks": raw_lm_fast}
        h2 = tracker._build_hand_data(det2, "Right", (480, 640), timestamp_sec=0.066)
        assert h2 is not None

        track = tracker._tracks["Right"]
        assert track.wrist_velocity > 0.80

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

        switched = tracker.switch_backend("tasks")
        assert tracker.active_backend_name in ("tasks", "solutions")

        switched_back = tracker.switch_backend("rtmpose")
        assert tracker.active_backend_name == "rtmpose"

    finally:
        tracker.close()
        backend.close()


