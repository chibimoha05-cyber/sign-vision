"""Data contracts and schema definitions for the sign language tracking pipeline.

Provides strongly-typed dataclasses and pure functions that define the canonical
144-dimensional feature vector structure, single-hand data containers, and the
frame-level result emitted by the tracking engine on every processed frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import numpy as np

LANDMARK_COUNT = 21
COORDINATES_PER_LANDMARK = 3
HAND_COORDINATE_DIM = LANDMARK_COUNT * COORDINATES_PER_LANDMARK
FINGER_COUNT = 5
HAND_FEATURE_DIM = HAND_COORDINATE_DIM + FINGER_COUNT
GLOBAL_KINEMATIC_DIM = 6

FEATURE_DIM_V1 = 128
FEATURE_DIM_V2 = 144
TOTAL_FEATURE_DIM = FEATURE_DIM_V2


@dataclass(slots=True)
class SingleHandData:
    """Tracking result, metric world landmarks, and biomechanical features for one hand."""

    handedness: Literal["Left", "Right"]
    confidence: float
    raw_landmarks: np.ndarray
    pixel_landmarks: np.ndarray
    normalized_landmarks: np.ndarray
    bounding_box: Tuple[int, int, int, int]
    world_landmarks: Optional[np.ndarray] = None
    finger_curls: Optional[np.ndarray] = None
    finger_splays: Optional[np.ndarray] = None
    pinch_distances: Optional[np.ndarray] = None
    is_coasting: bool = False
    binary_mask: int = 0
    tilt_angle_deg: float = 0.0
    confirmed_gesture: str = "none"
    gesture_state: str = "IDLE"

    def __post_init__(self) -> None:
        if self.raw_landmarks.shape != (LANDMARK_COUNT, 3):
            raise ValueError(f"raw_landmarks shape must be ({LANDMARK_COUNT}, 3), got {self.raw_landmarks.shape}")
        if self.pixel_landmarks.shape != (LANDMARK_COUNT, 2):
            raise ValueError(f"pixel_landmarks shape must be ({LANDMARK_COUNT}, 2), got {self.pixel_landmarks.shape}")
        if self.normalized_landmarks.shape != (LANDMARK_COUNT, 3):
            raise ValueError(f"normalized_landmarks shape must be ({LANDMARK_COUNT}, 3), got {self.normalized_landmarks.shape}")

        if self.world_landmarks is None:
            object.__setattr__(self, "world_landmarks", np.zeros((LANDMARK_COUNT, 3), dtype=np.float32))
        elif self.world_landmarks.shape != (LANDMARK_COUNT, 3):
            raise ValueError(f"world_landmarks shape must be ({LANDMARK_COUNT}, 3), got {self.world_landmarks.shape}")

        if self.finger_curls is None:
            object.__setattr__(self, "finger_curls", np.zeros(5, dtype=np.float32))
        elif self.finger_curls.shape != (5,):
            raise ValueError(f"finger_curls shape must be (5,), got {self.finger_curls.shape}")

        if self.finger_splays is None:
            object.__setattr__(self, "finger_splays", np.zeros(4, dtype=np.float32))
        elif self.finger_splays.shape != (4,):
            raise ValueError(f"finger_splays shape must be (4,), got {self.finger_splays.shape}")

        if self.pinch_distances is None:
            object.__setattr__(self, "pinch_distances", np.zeros(4, dtype=np.float32))
        elif self.pinch_distances.shape != (4,):
            raise ValueError(f"pinch_distances shape must be (4,), got {self.pinch_distances.shape}")


def build_feature_vector_v1(
    left_hand: Optional[SingleHandData],
    right_hand: Optional[SingleHandData],
) -> np.ndarray:
    """Constructs the legacy 128-float feature vector (v1 compatibility)."""
    feature_vec = np.zeros(FEATURE_DIM_V1, dtype=np.float32)
    feature_vec[0] = 1.0 if left_hand is not None else 0.0
    feature_vec[1] = 1.0 if right_hand is not None else 0.0

    if left_hand is not None:
        feature_vec[2:65] = left_hand.normalized_landmarks.astype(np.float32, copy=False).reshape(-1)

    if right_hand is not None:
        feature_vec[65:128] = right_hand.normalized_landmarks.astype(np.float32, copy=False).reshape(-1)

    return feature_vec


def build_feature_vector(
    left_hand: Optional[SingleHandData],
    right_hand: Optional[SingleHandData],
) -> np.ndarray:
    """Constructs the canonical 144-float biomechanical feature vector.

    Layout:
        [0..1]      2 values  — Hand presence flags (has_left, has_right)
        [2..64]    63 values  — Left hand normalized 3D coordinates (21 × 3)
        [65..69]    5 values  — Left hand finger curl ratios
        [70..132]  63 values  — Right hand normalized 3D coordinates (21 × 3)
        [133..137]  5 values  — Right hand finger curl ratios
        [138..140]  3 values  — Inter-finger splay angles (radians)
        [141..143]  3 values  — Key pinch distances (normalized)

    Missing hands are deterministically zero-padded. Total length: 144.
    """
    feature_vec = np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32)

    feature_vec[0] = 1.0 if left_hand is not None else 0.0
    feature_vec[1] = 1.0 if right_hand is not None else 0.0

    if left_hand is not None:
        feature_vec[2:65] = left_hand.normalized_landmarks.astype(np.float32, copy=False).reshape(-1)
        if left_hand.finger_curls is not None:
            feature_vec[65:70] = left_hand.finger_curls.astype(np.float32, copy=False)

    if right_hand is not None:
        feature_vec[70:133] = right_hand.normalized_landmarks.astype(np.float32, copy=False).reshape(-1)
        if right_hand.finger_curls is not None:
            feature_vec[133:138] = right_hand.finger_curls.astype(np.float32, copy=False)

    active_hand = right_hand if right_hand is not None else left_hand
    if active_hand is not None:
        if active_hand.finger_splays is not None and len(active_hand.finger_splays) >= 4:
            feature_vec[138:141] = active_hand.finger_splays[1:4]

        if active_hand.pinch_distances is not None and len(active_hand.pinch_distances) >= 4:
            feature_vec[141] = active_hand.pinch_distances[0]
            feature_vec[142] = active_hand.pinch_distances[1]
            feature_vec[143] = active_hand.pinch_distances[3]

    return feature_vec


@dataclass(slots=True)
class FrameTrackingResult:
    """Canonical frame-level output container from the hand tracking engine."""

    frame_index: int
    timestamp_ms: float
    inference_latency_ms: float
    left_hand: Optional[SingleHandData]
    right_hand: Optional[SingleHandData]
    presence_mask: Tuple[int, int]
    feature_vector: np.ndarray
    active_left_gesture: str = "none"
    active_right_gesture: str = "none"

    def __post_init__(self) -> None:
        if self.feature_vector.shape != (TOTAL_FEATURE_DIM,):
            raise ValueError(f"feature_vector shape must be ({TOTAL_FEATURE_DIM},), got {self.feature_vector.shape}")
        if self.feature_vector.dtype != np.float32:
            object.__setattr__(self, "feature_vector", self.feature_vector.astype(np.float32))

    @property
    def has_any_hand(self) -> bool:
        """True if at least one hand is detected."""
        return self.left_hand is not None or self.right_hand is not None

    @property
    def has_both_hands(self) -> bool:
        """True if both hands are detected."""
        return self.left_hand is not None and self.right_hand is not None
