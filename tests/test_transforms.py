"""Unit tests for spatial normalization and landmark stabilization."""

from __future__ import annotations

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sign_vision.schema import (
    LANDMARK_COUNT,
    TOTAL_FEATURE_DIM,
    SingleHandData,
    build_feature_vector,
    build_feature_vector_v1,
)
from sign_vision.transforms import (
    HandStabilizer,
    compute_bounding_box,
    normalize_landmarks,
)


@pytest.fixture
def synthetic_hand_skeleton() -> np.ndarray:
    """Generates a realistic 21-landmark 3D hand skeleton in normalized camera coordinates."""
    pts = np.zeros((21, 3), dtype=np.float64)
    pts[0] = [0.50, 0.70, 0.00]

    pts[1] = [0.46, 0.65, -0.01]
    pts[2] = [0.43, 0.60, -0.02]
    pts[3] = [0.40, 0.56, -0.03]
    pts[4] = [0.38, 0.52, -0.04]

    pts[5] = [0.47, 0.55, -0.02]
    pts[6] = [0.46, 0.48, -0.03]
    pts[7] = [0.45, 0.43, -0.04]
    pts[8] = [0.44, 0.38, -0.05]

    pts[9] = [0.50, 0.54, -0.02]
    pts[10] = [0.50, 0.46, -0.03]
    pts[11] = [0.50, 0.40, -0.04]
    pts[12] = [0.50, 0.35, -0.05]

    pts[13] = [0.53, 0.55, -0.02]
    pts[14] = [0.54, 0.48, -0.03]
    pts[15] = [0.55, 0.43, -0.04]
    pts[16] = [0.56, 0.38, -0.05]

    pts[17] = [0.56, 0.58, -0.01]
    pts[18] = [0.58, 0.53, -0.02]
    pts[19] = [0.60, 0.48, -0.03]
    pts[20] = [0.62, 0.44, -0.04]

    return pts


def test_translation_invariance(synthetic_hand_skeleton: np.ndarray) -> None:
    """Shifts all landmarks by (+150, -80, +30) and verifies exact mathematical identity."""
    base_normalized = normalize_landmarks(synthetic_hand_skeleton)

    shift_vector = np.array([150.0, -80.0, 30.0], dtype=np.float64)
    shifted_skeleton = synthetic_hand_skeleton + shift_vector

    shifted_normalized = normalize_landmarks(shifted_skeleton)

    np.testing.assert_allclose(
        shifted_normalized,
        base_normalized,
        atol=1e-5,
        err_msg="Translation invariance violated: shifted skeleton produced differing normalized output.",
    )


def test_scale_invariance(synthetic_hand_skeleton: np.ndarray) -> None:
    """Scales skeleton by 3.5x (simulating moving closer to camera) and verifies exact identity."""
    base_normalized = normalize_landmarks(synthetic_hand_skeleton)

    scaled_skeleton = synthetic_hand_skeleton * 3.5

    scaled_normalized = normalize_landmarks(scaled_skeleton)

    np.testing.assert_allclose(
        scaled_normalized,
        base_normalized,
        atol=1e-5,
        err_msg="Scale invariance violated: scaled skeleton produced differing normalized output.",
    )


def test_combined_translation_and_scale_invariance(synthetic_hand_skeleton: np.ndarray) -> None:
    """Applies simultaneous arbitrary translation and scaling."""
    base_normalized = normalize_landmarks(synthetic_hand_skeleton)

    transformed_skeleton = (synthetic_hand_skeleton * 0.25) + np.array([-42.5, 108.3, -19.7], dtype=np.float64)
    transformed_normalized = normalize_landmarks(transformed_skeleton)

    np.testing.assert_allclose(
        transformed_normalized,
        base_normalized,
        atol=1e-5,
        err_msg="Combined translation and scale invariance failed.",
    )


def test_normalized_landmarks_bounds(synthetic_hand_skeleton: np.ndarray) -> None:
    """Verifies that all normalized landmarks are strictly within [-1.0, 1.0] and wrist is (0, 0, 0)."""
    norm = normalize_landmarks(synthetic_hand_skeleton)

    assert norm.shape == (21, 3)
    assert norm.dtype == np.float32

    np.testing.assert_allclose(norm[0], [0.0, 0.0, 0.0], atol=1e-6)

    distances = np.linalg.norm(norm, axis=1)
    assert np.all(distances <= 1.0 + 1e-6), f"Landmarks exceeded unit sphere: max={np.max(distances)}"

    assert np.all(norm >= -1.0 - 1e-6)
    assert np.all(norm <= 1.0 + 1e-6)


def test_degenerate_zero_skeleton() -> None:
    """Verifies that a degenerate zero-distance skeleton handles division by zero safely via epsilon."""
    zero_skeleton = np.zeros((21, 3), dtype=np.float32)
    norm = normalize_landmarks(zero_skeleton)

    assert not np.any(np.isnan(norm))
    assert not np.any(np.isinf(norm))
    np.testing.assert_allclose(norm, np.zeros((21, 3), dtype=np.float32))


def test_bounding_box_computation() -> None:
    """Verifies bounding box calculation and image clamping."""
    px_coords = np.array([[100, 100], [200, 250], [150, 180]], dtype=np.int32)
    px_landmarks = np.zeros((21, 2), dtype=np.int32)
    px_landmarks[:3] = px_coords

    bbox = compute_bounding_box(px_landmarks, (480, 640), padding_ratio=0.1)
    x_min, y_min, x_max, y_max = bbox

    assert 0 <= x_min <= x_max < 640
    assert 0 <= y_min <= y_max < 480


def test_feature_vector_contracts(synthetic_hand_skeleton: np.ndarray) -> None:
    """Validates the strict 144-float feature vector contract across 0-hand, 1-hand, and 2-hand cases."""
    norm = normalize_landmarks(synthetic_hand_skeleton)
    px = np.zeros((21, 2), dtype=np.int32)
    bbox = (0, 0, 100, 100)

    left_hand = SingleHandData(
        handedness="Left",
        confidence=0.95,
        raw_landmarks=synthetic_hand_skeleton,
        pixel_landmarks=px,
        normalized_landmarks=norm,
        bounding_box=bbox,
    )

    right_hand = SingleHandData(
        handedness="Right",
        confidence=0.91,
        raw_landmarks=synthetic_hand_skeleton,
        pixel_landmarks=px,
        normalized_landmarks=norm,
        bounding_box=bbox,
    )

    vec_none = build_feature_vector(None, None)
    assert vec_none.shape == (TOTAL_FEATURE_DIM,)
    assert vec_none.dtype == np.float32
    assert not np.any(np.isnan(vec_none))
    assert not np.any(np.isinf(vec_none))
    assert vec_none[0] == 0.0
    assert vec_none[1] == 0.0
    np.testing.assert_allclose(vec_none[2:], 0.0)

    vec_left = build_feature_vector(left_hand, None)
    assert vec_left.shape == (TOTAL_FEATURE_DIM,)
    assert vec_left[0] == 1.0
    assert vec_left[1] == 0.0
    np.testing.assert_allclose(vec_left[2:65], norm.reshape(-1))
    np.testing.assert_allclose(vec_left[70:138], 0.0)

    vec_right = build_feature_vector(None, right_hand)
    assert vec_right.shape == (TOTAL_FEATURE_DIM,)
    assert vec_right[0] == 0.0
    assert vec_right[1] == 1.0
    np.testing.assert_allclose(vec_right[2:70], 0.0)
    np.testing.assert_allclose(vec_right[70:133], norm.reshape(-1))

    vec_both = build_feature_vector(left_hand, right_hand)
    assert vec_both.shape == (TOTAL_FEATURE_DIM,)
    assert vec_both[0] == 1.0
    assert vec_both[1] == 1.0
    np.testing.assert_allclose(vec_both[2:65], norm.reshape(-1))
    np.testing.assert_allclose(vec_both[70:133], norm.reshape(-1))

    vec_v1 = build_feature_vector_v1(left_hand, right_hand)
    assert vec_v1.shape == (128,)


def test_adaptive_ema_stabilizer(synthetic_hand_skeleton: np.ndarray) -> None:
    """Verifies that HandStabilizer adapts between slow and fast movement and resets properly."""
    norm = normalize_landmarks(synthetic_hand_skeleton)
    stabilizer = HandStabilizer(alpha_slow=0.3, alpha_fast=0.9, v_low=0.01, v_high=0.08, max_missing_frames=3)

    f1 = stabilizer.filter(norm, "Right")
    np.testing.assert_allclose(f1, norm)

    jitter = np.zeros_like(norm)
    jitter[5:] = 0.005
    f2_input = norm + jitter
    f2 = stabilizer.filter(f2_input, "Right")
    diff_smoothed = np.linalg.norm(f2 - norm)
    diff_input = np.linalg.norm(f2_input - norm)
    assert diff_smoothed < diff_input * 0.7, "EMA failed to smooth low-velocity jitter."

    fast_shift = np.zeros_like(norm)
    fast_shift[5:] = 0.50
    f3_input = norm + fast_shift
    f3 = stabilizer.filter(f3_input, "Right")
    assert np.linalg.norm(f3 - f3_input) < 0.25 * np.linalg.norm(f3_input - norm)

    stabilizer.register_missing("Right")
    stabilizer.register_missing("Right")
    stabilizer.register_missing("Right")

    f4 = stabilizer.filter(norm, "Right")
    np.testing.assert_allclose(f4, norm)


def test_one_euro_filter_stationary_jitter_suppression(synthetic_hand_skeleton: np.ndarray) -> None:
    """Verifies that OneEuroFilter dampens high-frequency sensor noise on stationary hand."""
    from sign_vision.transforms import OneEuroFilter

    filt = OneEuroFilter(min_cutoff=1.2, beta=0.04, d_cutoff=1.0)
    base = synthetic_hand_skeleton.astype(np.float32)

    np.random.seed(42)
    raw_jitter_diffs = []
    filtered_jitter_diffs = []

    t = 0.0
    dt = 0.033

    for _ in range(40):
        t += dt
        noise = np.random.normal(0.0, 0.015, size=base.shape).astype(np.float32)
        noisy_frame = base + noise
        filtered_frame = filt.filter(noisy_frame, t)

        raw_jitter_diffs.append(np.linalg.norm(noisy_frame - base))
        filtered_jitter_diffs.append(np.linalg.norm(filtered_frame - base))

    avg_raw_jitter = np.mean(raw_jitter_diffs[5:])
    avg_filt_jitter = np.mean(filtered_jitter_diffs[5:])

    assert avg_filt_jitter < avg_raw_jitter * 0.45, (
        f"OneEuroFilter failed to suppress jitter: raw={avg_raw_jitter:.4f}, filtered={avg_filt_jitter:.4f}"
    )


def test_one_euro_filter_fast_movement_responsiveness(synthetic_hand_skeleton: np.ndarray) -> None:
    """Verifies that OneEuroFilter dynamically increases cutoff to avoid lag during fast gestures."""
    from sign_vision.transforms import OneEuroFilter

    filt = OneEuroFilter(min_cutoff=1.2, beta=0.04, d_cutoff=1.0)
    base = synthetic_hand_skeleton.astype(np.float32)

    t = 0.0
    dt = 0.033
    for _ in range(5):
        t += dt
        filt.filter(base, t)

    target = base + 0.40
    t += dt
    jump_f1 = filt.filter(target, t)

    step_taken = np.mean(np.abs(jump_f1 - base))
    assert step_taken > 0.05, f"Filter was too sluggish on step jump: step_taken={step_taken:.4f}"

    for _ in range(4):
        t += dt
        jump_final = filt.filter(target, t)

    mean_residual_error = np.mean(np.abs(jump_final - target))
    assert mean_residual_error < 0.12, f"OneEuroFilter failed to converge quickly: residual={mean_residual_error:.4f}"


def test_hand_stabilizer_filter_raw(synthetic_hand_skeleton: np.ndarray) -> None:
    """Verifies that HandStabilizer filter_raw applies One-Euro smoothing at source."""
    stabilizer = HandStabilizer(mode="one_euro")
    raw = synthetic_hand_skeleton.astype(np.float32)

    out1 = stabilizer.filter_raw(raw, "Left", 0.0)
    np.testing.assert_allclose(out1, raw)

    out2 = stabilizer.filter_raw(raw + 0.005, "Left", 0.033)
    assert out2.shape == (21, 3)
    assert out2.dtype == np.float32
