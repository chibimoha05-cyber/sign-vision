"""Unit tests for Gesture Engine, Arbiters, Binary Masks, and Temporal State Machines."""

import numpy as np
import pytest

from sign_vision.gesture_engine import (
    GestureArbiter,
    GestureName,
    GestureState,
    HandGestureStateMachine,
)
from sign_vision.kinematics import HandKinematicsEngine


def test_binary_mask_computation():
    """Validates 5-bit binary bitmask calculation for canonical hand poses."""
    engine = HandKinematicsEngine()

    # Synthetic neutral open hand
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[0] = [0.5, 0.8, 0.0]  # Wrist
    landmarks[9] = [0.5, 0.6, 0.0]  # Middle MCP

    # All open: curls = 1.0 -> mask = 31 (0b11111)
    open_curls = np.ones(5, dtype=np.float32)
    mask_open, states_open = engine.compute_binary_mask(landmarks, open_curls)
    assert mask_open == 31
    assert all(states_open.values())

    # All curled: curls = 0.0 -> mask = 0 (0b00000)
    curled_curls = np.zeros(5, dtype=np.float32)
    mask_fist, states_fist = engine.compute_binary_mask(landmarks, curled_curls)
    assert mask_fist == 0
    assert not any(states_fist.values())

    # Peace sign: Index + Middle open (8 + 4 = 12)
    peace_curls = np.array([0.1, 0.9, 0.9, 0.1, 0.1], dtype=np.float32)
    mask_peace, _ = engine.compute_binary_mask(landmarks, peace_curls)
    assert mask_peace == 12


def test_hand_tilt_angle_calculation():
    """Validates hand tilt angle calculation across orientations."""
    engine = HandKinematicsEngine()

    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[0] = [0.5, 0.8, 0.0]  # Wrist

    # Upright: tips have smaller Y (higher on screen) than wrist
    landmarks[8] = [0.5, 0.4, 0.0]
    landmarks[12] = [0.5, 0.35, 0.0]
    landmarks[16] = [0.5, 0.4, 0.0]
    angle_up = engine.compute_hand_tilt_angle(landmarks)
    assert abs(angle_up) < 5.0  # ~0 degrees

    # Pointing right: tips have larger X than wrist
    landmarks[8] = [0.8, 0.8, 0.0]
    landmarks[12] = [0.85, 0.8, 0.0]
    landmarks[16] = [0.8, 0.8, 0.0]
    angle_right = engine.compute_hand_tilt_angle(landmarks)
    assert 80.0 < angle_right < 100.0  # ~+90 degrees

    # Pointing left: tips have smaller X than wrist
    landmarks[8] = [0.2, 0.8, 0.0]
    landmarks[12] = [0.15, 0.8, 0.0]
    landmarks[16] = [0.2, 0.8, 0.0]
    angle_left = engine.compute_hand_tilt_angle(landmarks)
    assert -100.0 < angle_left < -80.0  # ~-90 degrees


def test_gesture_arbiter_recognition():
    """Validates pure gesture recognizers and score-based arbitration."""
    arbiter = GestureArbiter()
    pinches_default = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    # 1. Fist
    fist_curls = np.zeros(5, dtype=np.float32)
    winner_fist = arbiter.arbitrate(
        binary_mask=0,
        finger_curls=fist_curls,
        pinch_distances=pinches_default,
        tilt_angle_deg=0.0,
    )
    assert winner_fist.valid
    assert winner_fist.gesture == GestureName.FIST

    # 2. Open Palm
    open_curls = np.ones(5, dtype=np.float32)
    winner_open = arbiter.arbitrate(
        binary_mask=31,
        finger_curls=open_curls,
        pinch_distances=pinches_default,
        tilt_angle_deg=0.0,
    )
    assert winner_open.valid
    assert winner_open.gesture == GestureName.OPEN_PALM

    # 3. Peace V
    peace_curls = np.array([0.1, 0.9, 0.9, 0.1, 0.1], dtype=np.float32)
    winner_peace = arbiter.arbitrate(
        binary_mask=12,
        finger_curls=peace_curls,
        pinch_distances=pinches_default,
        tilt_angle_deg=0.0,
    )
    assert winner_peace.valid
    assert winner_peace.gesture == GestureName.PEACE_V

    # 4. OK Pinch (Thumb touching Index tip)
    ok_pinches = np.array([0.08, 0.6, 0.6, 0.6], dtype=np.float32)
    ok_curls = np.array([0.3, 0.3, 0.9, 0.9, 0.9], dtype=np.float32)
    winner_ok = arbiter.arbitrate(
        binary_mask=7,
        finger_curls=ok_curls,
        pinch_distances=ok_pinches,
        tilt_angle_deg=0.0,
    )
    assert winner_ok.valid
    assert winner_ok.gesture == GestureName.OK_PINCH


def test_gesture_state_machine_transitions():
    """Tests IDLE -> CANDIDATE -> ACTIVE confirmation and debounced release."""
    sm = HandGestureStateMachine(confirm_frames=2, release_frames=2)
    pinches = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    fist_curls = np.zeros(5, dtype=np.float32)

    assert sm.state == GestureState.IDLE

    # Frame 1: Fist detected -> transitions to CANDIDATE
    g1, state1 = sm.process(binary_mask=0, finger_curls=fist_curls, pinch_distances=pinches, tilt_angle_deg=0.0)
    assert state1 == GestureState.CANDIDATE.value
    assert g1 == GestureName.NONE

    # Frame 2: Fist confirmed (confirm_frames=2) -> transitions to ACTIVE
    g2, state2 = sm.process(binary_mask=0, finger_curls=fist_curls, pinch_distances=pinches, tilt_angle_deg=0.0)
    assert state2 == GestureState.ACTIVE.value
    assert g2 == GestureName.FIST

    # Frame 3: Dropout/flicker -> enters RELEASING, but still reports active gesture
    empty_curls = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    g3, state3 = sm.process(binary_mask=10, finger_curls=empty_curls, pinch_distances=pinches, tilt_angle_deg=0.0)
    assert state3 == GestureState.RELEASING.value
    assert g3 == GestureName.FIST  # Debounced retention

    # Frame 4: Second frame of absence (release_frames=2) -> transitions to IDLE
    g4, state4 = sm.process(binary_mask=10, finger_curls=empty_curls, pinch_distances=pinches, tilt_angle_deg=0.0)
    assert state4 == GestureState.IDLE.value
    assert g4 == GestureName.NONE


def test_instant_priority_preemption():
    """Validates that a higher-priority gesture instantly interrupts an active lower-priority gesture."""
    sm = HandGestureStateMachine(confirm_frames=1, release_frames=2)
    pinches = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    # Activate Open Palm (Priority 50)
    open_curls = np.ones(5, dtype=np.float32)
    g1, state1 = sm.process(binary_mask=31, finger_curls=open_curls, pinch_distances=pinches, tilt_angle_deg=0.0)
    assert state1 == GestureState.ACTIVE.value
    assert g1 == GestureName.OPEN_PALM

    # Suddenly user pinches (OK Pinch, Priority 95)
    # Must immediately preempt without going through IDLE
    ok_pinches = np.array([0.05, 0.6, 0.6, 0.6], dtype=np.float32)
    ok_curls = np.array([0.3, 0.3, 0.9, 0.9, 0.9], dtype=np.float32)
    g2, state2 = sm.process(binary_mask=7, finger_curls=ok_curls, pinch_distances=ok_pinches, tilt_angle_deg=0.0)
    assert state2 == GestureState.ACTIVE.value
    assert g2 == GestureName.OK_PINCH
