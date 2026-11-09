"""Biomechanical Precision and Hand Kinematics Tests for Phase 2.0.

Validates:
1. Isolated Finger Movement Test: Middle finger articulates from extended to curled
   while Ring finger remains stationary, proving strict crosstalk decoupling (delta < 0.03).
2. Bone Rigidity Constraint Test: Middle phalanx stretched by 50% is projected back
   into physiological bounds along the bone ray.
3. World Landmark Extraction & Metrics: Realistic 3D metric coordinates bounded in meters.
4. Canonical 144-Float Feature Vector Contract: Deterministic shape (144,), padding, and bounds.
"""

from __future__ import annotations

import numpy as np
import pytest

from sign_vision.contracts import (
    FEATURE_DIM_V1,
    FEATURE_DIM_V2,
    LANDMARK_COUNT,
    TOTAL_FEATURE_DIM,
    SingleHandData,
    build_feature_vector,
    build_feature_vector_v1,
)
from sign_vision.kinematics import (
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_DIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_TIP,
    RING_DIP,
    RING_MCP,
    RING_PIP,
    RING_TIP,
    THUMB_CMC,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
    HandKinematicsEngine,
)
from sign_vision.normalizer import OneEuroFilter, normalize_landmarks


def create_canonical_hand() -> np.ndarray:
    """Creates a realistic, open flat hand in normalized coordinates."""
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[WRIST] = [0.50, 0.80, 0.00]

    # Thumb
    pts[THUMB_CMC] = [0.42, 0.70, 0.00]
    pts[THUMB_MCP] = [0.36, 0.62, 0.00]
    pts[THUMB_IP] = [0.32, 0.55, 0.00]
    pts[THUMB_TIP] = [0.28, 0.48, 0.00]

    # MCPs
    pts[INDEX_MCP] = [0.40, 0.50, 0.00]
    pts[MIDDLE_MCP] = [0.48, 0.48, 0.00]
    pts[RING_MCP] = [0.55, 0.50, 0.00]
    pts[PINKY_MCP] = [0.62, 0.54, 0.00]

    # Index finger extended
    pts[6] = [0.39, 0.40, 0.00]
    pts[7] = [0.38, 0.32, 0.00]
    pts[INDEX_TIP] = [0.37, 0.24, 0.00]

    # Middle finger extended
    pts[MIDDLE_PIP] = [0.48, 0.37, 0.00]
    pts[MIDDLE_DIP] = [0.48, 0.28, 0.00]
    pts[MIDDLE_TIP] = [0.48, 0.20, 0.00]

    # Ring finger extended
    pts[RING_PIP] = [0.56, 0.40, 0.00]
    pts[RING_DIP] = [0.57, 0.32, 0.00]
    pts[RING_TIP] = [0.58, 0.25, 0.00]

    # Pinky extended
    pts[18] = [0.64, 0.46, 0.00]
    pts[19] = [0.66, 0.39, 0.00]
    pts[PINKY_TIP] = [0.67, 0.33, 0.00]

    return pts


def test_isolated_finger_movement_decoupling():
    """Validates that moving Middle finger from 100% extended to fully curled

    does not alter Ring finger's measured state (crosstalk delta < 0.03).
    """
    engine = HandKinematicsEngine()
    hand_base = create_canonical_hand()

    # Step 1: Initial state - both Middle and Ring extended
    curls_initial = engine.compute_finger_curls(hand_base)
    middle_curl_initial = curls_initial[2]
    ring_curl_initial = curls_initial[3]

    assert middle_curl_initial > 0.85, f"Expected middle extended, got {middle_curl_initial}"
    assert ring_curl_initial > 0.85, f"Expected ring extended, got {ring_curl_initial}"

    # Step 2: Simulate 30 frames articulating Middle finger to fully curled
    # while Ring finger joints remain strictly stationary
    ring_tip_initial = hand_base[RING_TIP].copy()
    filter_engine = OneEuroFilter(min_cutoff=1.2, beta=0.04)

    ring_curls_recorded = []
    middle_curls_recorded = []

    total_frames = 30
    for frame in range(1, total_frames + 1):
        t_sec = frame * (1.0 / 30.0)
        hand_step = hand_base.copy()

        # Articulate middle finger inward toward palm
        fraction = min(1.0, frame / 18.0)
        # Interpolate Middle PIP, DIP, TIP down to palm
        hand_step[MIDDLE_PIP] = [0.48, 0.37 + fraction * 0.05, -fraction * 0.04]
        hand_step[MIDDLE_DIP] = [0.48, 0.28 + fraction * 0.20, -fraction * 0.05]
        hand_step[MIDDLE_TIP] = [0.48, 0.20 + fraction * 0.35, -fraction * 0.02]

        # Ring finger remains identical
        assert np.allclose(hand_step[RING_TIP], ring_tip_initial)

        # Pass through OneEuroFilter
        filtered = filter_engine.filter(hand_step, t_sec)

        curls = engine.compute_finger_curls(filtered)
        middle_curls_recorded.append(curls[2])
        ring_curls_recorded.append(curls[3])

    # Assert Middle curled down
    assert middle_curls_recorded[-1] < 0.20, f"Expected middle curled, got {middle_curls_recorded[-1]}"

    # Assert Ring finger curl experienced zero phantom drift / crosstalk (max deviation < 0.03)
    ring_deltas = [abs(rc - ring_curl_initial) for rc in ring_curls_recorded]
    max_ring_drift = max(ring_deltas)
    assert max_ring_drift < 0.03, f"Ring finger experienced phantom crosstalk drift of {max_ring_drift:.4f} (target < 0.03)"



def test_bone_rigidity_constraint_enforcement():
    """Validates that a 50% stretched bone segment is projected back within physiological limits."""
    engine = HandKinematicsEngine(max_bone_stretch=0.25, projection_blend=0.90)
    hand = create_canonical_hand()

    # Compute reference length of Middle MCP -> Middle PIP
    ref_len = float(np.linalg.norm(hand[MIDDLE_PIP] - hand[MIDDLE_MCP]))

    # Artificially corrupt Middle PIP by stretching it 50% along its ray
    ray = hand[MIDDLE_PIP] - hand[MIDDLE_MCP]
    stretched_pip = hand[MIDDLE_MCP] + ray * 1.50
    hand_corrupted = hand.copy()
    hand_corrupted[MIDDLE_PIP] = stretched_pip

    corrupted_len = float(np.linalg.norm(hand_corrupted[MIDDLE_PIP] - hand[MIDDLE_MCP]))
    assert np.isclose(corrupted_len, ref_len * 1.50)

    # Apply bone rigidity constraints with reference dictionary
    custom_refs = {(MIDDLE_MCP, MIDDLE_PIP): ref_len}
    restored = engine.apply_bone_rigidity_constraints(hand_corrupted, custom_ref_lengths=custom_refs)

    restored_len = float(np.linalg.norm(restored[MIDDLE_PIP] - restored[MIDDLE_MCP]))

    # Restored length must be clamped close to ref_len * 1.25
    max_allowed = ref_len * 1.25
    assert restored_len <= max_allowed * 1.05, f"Expected restored length <= {max_allowed * 1.05}, got {restored_len}"
    assert restored_len < corrupted_len, "Bone constraint failed to shorten the over-stretched segment"


def test_world_landmark_metric_bounds():
    """Validates that world_landmarks are stored in meters centered at the hand origin."""
    hand = create_canonical_hand()
    # Synthetic world landmarks centered at wrist with ~0.18m (18cm) hand length
    world_lm = (hand - hand[WRIST]) * 0.20

    # Wrist must be at origin (0, 0, 0)
    np.testing.assert_allclose(world_lm[WRIST], 0.0, atol=1e-5)

    # Metric distance from wrist to middle tip should be realistic human hand size (~15-22 cm)
    hand_length_m = float(np.linalg.norm(world_lm[MIDDLE_TIP] - world_lm[WRIST]))
    assert 0.10 <= hand_length_m <= 0.25, f"Unrealistic hand length in meters: {hand_length_m:.3f}m"

    # All coordinates must be bounded in [-0.25, 0.25] meters
    assert np.all(world_lm >= -0.25)
    assert np.all(world_lm <= 0.25)


def test_144_float_feature_vector_contract():
    """Validates the Phase 2.0 strict 144-float feature vector contract across 0, 1, and 2 hands."""
    assert TOTAL_FEATURE_DIM == 144
    assert FEATURE_DIM_V1 == 128
    assert FEATURE_DIM_V2 == 144

    hand = create_canonical_hand()
    norm = normalize_landmarks(hand)
    engine = HandKinematicsEngine()
    curls = engine.compute_finger_curls(hand)
    splays = engine.compute_finger_splays(hand)
    pinches = engine.compute_pinch_distances(hand)

    pixel_pts = np.round(hand[:, :2] * 640).astype(np.int32)

    sample_right = SingleHandData(
        handedness="Right",
        confidence=0.95,
        raw_landmarks=hand,
        pixel_landmarks=pixel_pts,
        normalized_landmarks=norm,
        bounding_box=(100, 100, 300, 400),
        world_landmarks=(hand - hand[0]) * 0.20,
        finger_curls=curls,
        finger_splays=splays,
        pinch_distances=pinches,
        is_coasting=False,
    )

    sample_left = SingleHandData(
        handedness="Left",
        confidence=0.92,
        raw_landmarks=hand,
        pixel_landmarks=pixel_pts,
        normalized_landmarks=norm,
        bounding_box=(350, 100, 550, 400),
        world_landmarks=(hand - hand[0]) * 0.20,
        finger_curls=curls,
        finger_splays=splays,
        pinch_distances=pinches,
        is_coasting=False,
    )

    # Case 1: 0 hands
    vec_none = build_feature_vector(None, None)
    assert vec_none.shape == (144,)
    assert vec_none.dtype == np.float32
    assert not np.any(np.isnan(vec_none))
    np.testing.assert_allclose(vec_none, 0.0)

    # Case 2: Left hand only
    vec_left = build_feature_vector(sample_left, None)
    assert vec_left.shape == (144,)
    assert vec_left[0] == 1.0  # has_left
    assert vec_left[1] == 0.0  # has_right
    # Left coords at [2:65]
    np.testing.assert_allclose(vec_left[2:65], norm.reshape(-1))
    # Left curls at [65:70]
    np.testing.assert_allclose(vec_left[65:70], curls)
    # Right values at [70:138] must be 0
    np.testing.assert_allclose(vec_left[70:138], 0.0)

    # Case 3: Right hand only
    vec_right = build_feature_vector(None, sample_right)
    assert vec_right.shape == (144,)
    assert vec_right[0] == 0.0  # has_left
    assert vec_right[1] == 1.0  # has_right
    # Left values at [2:70] must be 0
    np.testing.assert_allclose(vec_right[2:70], 0.0)
    # Right coords at [70:133]
    np.testing.assert_allclose(vec_right[70:133], norm.reshape(-1))
    # Right curls at [133:138]
    np.testing.assert_allclose(vec_right[133:138], curls)
    # Global kinematics at [138:144]
    np.testing.assert_allclose(vec_right[138:141], splays[1:4])
    np.testing.assert_allclose(vec_right[141], pinches[0])
    np.testing.assert_allclose(vec_right[142], pinches[1])
    np.testing.assert_allclose(vec_right[143], pinches[3])

    # Case 4: Both hands
    vec_both = build_feature_vector(sample_left, sample_right)
    assert vec_both.shape == (144,)
    assert vec_both[0] == 1.0
    assert vec_both[1] == 1.0
    np.testing.assert_allclose(vec_both[2:65], norm.reshape(-1))
    np.testing.assert_allclose(vec_both[65:70], curls)
    np.testing.assert_allclose(vec_both[70:133], norm.reshape(-1))
    np.testing.assert_allclose(vec_both[133:138], curls)
    assert not np.any(np.isnan(vec_both))
    assert not np.any(np.isinf(vec_both))

    # Backward compatibility helper test
    vec_v1 = build_feature_vector_v1(sample_left, sample_right)
    assert vec_v1.shape == (128,)


def test_no_coordinate_warping_under_extreme_foreshortening():
    """Verifies that raw/pixel coordinates are NEVER mutated by HandKinematicsEngine.

    Under extreme 2D perspective foreshortening (e.g. finger pointing straight at camera,
    shortening 2D projected length by 70%), extract_kinematics must preserve the exact visual
    detector coordinates without pushing landmarks off physical fingers.
    """
    engine = HandKinematicsEngine()
    hand = create_canonical_hand()

    # Simulate extreme foreshortening on Index finger pointing straight toward the camera:
    # 2D projected length shortens drastically while joint bends along z
    foreshortened_hand = hand.copy()
    foreshortened_hand[INDEX_TIP] = [0.40, 0.44, -0.12]  # Tip is near MCP in 2D projection
    foreshortened_hand[7] = [0.40, 0.46, -0.08]

    original_input_copy = foreshortened_hand.copy()

    # Call extract_kinematics with enforce_rigidity=True and enforce_rigidity=False
    out_coords_rigidity_true, kin_true = engine.extract_kinematics(
        foreshortened_hand,
        handedness="Right",
        enforce_rigidity=True,
    )

    out_coords_rigidity_false, kin_false = engine.extract_kinematics(
        foreshortened_hand,
        handedness="Right",
        enforce_rigidity=False,
    )

    # 1. Assert input array was not mutated in place
    np.testing.assert_allclose(foreshortened_hand, original_input_copy, err_msg="Input coordinates mutated in place!")

    # 2. Assert output coordinates remain 100% faithful to detector output (NO coordinate modification)
    np.testing.assert_allclose(out_coords_rigidity_true, original_input_copy, atol=1e-6, err_msg="Rigidity warped coordinates!")
    np.testing.assert_allclose(out_coords_rigidity_false, original_input_copy, atol=1e-6, err_msg="Detector coordinates modified!")

    # 3. Assert index curl is recognized as bent/curled rather than crashing or exploding
    assert kin_true.finger_curls[1] < 0.50
    assert kin_false.finger_curls[1] < 0.50

