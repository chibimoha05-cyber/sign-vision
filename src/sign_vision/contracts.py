"""Data contracts defining strict schemas for the Sign Language Translation Pipeline.

These contracts provide deterministic, strongly-typed data structures for:
- Team Member 2: Machine Learning Trainer & Classifier
- Team Member 3: Dataset Collection & Curation
- Team Member 4: Translation Engine / NLP / Gesture State Machine
- Team Member 5: User Interface & Visual Feedback HUD
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import numpy as np

# Landmark count for MediaPipe hands
LANDMARK_COUNT = 21
COORDINATES_PER_LANDMARK = 3
HAND_COORDINATE_DIM = LANDMARK_COUNT * COORDINATES_PER_LANDMARK  # 63
FINGER_COUNT = 5
HAND_FEATURE_DIM = HAND_COORDINATE_DIM + FINGER_COUNT  # 68 (63 coords + 5 curl ratios)
GLOBAL_KINEMATIC_DIM = 6  # 3 splay angles + 3 key pinch distances

FEATURE_DIM_V1 = 128  # Legacy Phase 1 contract (2 + 63 + 63)
FEATURE_DIM_V2 = 144  # Phase 2.0 Biomechanical contract (2 + 68 + 68 + 6)
TOTAL_FEATURE_DIM = FEATURE_DIM_V2  # 144


@dataclass(slots=True)
class SingleHandData:
    """Detailed tracking, metric world landmarks, and biomechanical features for a detected hand."""

    handedness: Literal["Left", "Right"]
    confidence: float
    raw_landmarks: np.ndarray  # shape (21, 3), float32 (normalized image coordinates [0, 1])
    pixel_landmarks: np.ndarray  # shape (21, 2), int32 (pixel coordinates [px_x, px_y])
    normalized_landmarks: np.ndarray  # shape (21, 3), float32 (wrist-centered, scale-invariant [-1, 1])
    bounding_box: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max) in pixel coordinates
    world_landmarks: Optional[np.ndarray] = None  # shape (21, 3), float32 (metric 3D Cartesian in meters)
    finger_curls: Optional[np.ndarray] = None  # shape (5,), float32 in [0.0, 1.0] [Thumb, Index, Mid, Ring, Pinky]
    finger_splays: Optional[np.ndarray] = None  # shape (4,), float32 in radians [T-I, I-M, M-R, R-P]
    pinch_distances: Optional[np.ndarray] = None  # shape (4,), float32 normalized [T-I, T-M, T-R, T-P]
    is_coasting: bool = False  # True when bridging a 1-3 frame detection dropout
    binary_mask: int = 0  # 5-bit finger mask (0..31: Thumb=16, Index=8, Mid=4, Ring=2, Pinky=1)
    tilt_angle_deg: float = 0.0  # Palm tilt angle in degrees (-180..180, 0=upright, +90=right, -90=left)
    confirmed_gesture: str = "none"  # Temporal debounced gesture name
    gesture_state: str = "IDLE"  # State machine status (IDLE, CANDIDATE, CONFIRMED, ACTIVE, RELEASING)

    def __post_init__(self) -> None:
        if self.raw_landmarks.shape != (LANDMARK_COUNT, 3):
            raise ValueError(f"raw_landmarks shape must be ({LANDMARK_COUNT}, 3), got {self.raw_landmarks.shape}")
        if self.pixel_landmarks.shape != (LANDMARK_COUNT, 2):
            raise ValueError(f"pixel_landmarks shape must be ({LANDMARK_COUNT}, 2), got {self.pixel_landmarks.shape}")
        if self.normalized_landmarks.shape != (LANDMARK_COUNT, 3):
            raise ValueError(f"normalized_landmarks shape must be ({LANDMARK_COUNT}, 3), got {self.normalized_landmarks.shape}")

        # Set deterministic defaults if optional kinematic fields are omitted
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
    """Constructs the legacy 128-float feature vector (Phase 1)."""
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
    """Constructs the canonical 144-float biomechanical feature vector (Phase 2.0).

    Structure:
    - Indices 0..1: [has_left, has_right] (2 values)
    - Left Hand (68 values):
      - Indices 2..64: Flattened normalized landmarks (21 x 3 = 63 values)
      - Indices 65..69: Finger curl ratios [Thumb, Index, Mid, Ring, Pinky] (5 values)
    - Right Hand (68 values):
      - Indices 70..132: Flattened normalized landmarks (21 x 3 = 63 values)
      - Indices 133..137: Finger curl ratios [Thumb, Index, Mid, Ring, Pinky] (5 values)
    - Global Kinematics (6 values):
      - Indices 138..140: 3 inter-finger splay angles [Index-Mid, Mid-Ring, Ring-Pinky]
      - Indices 141..143: 3 key pinch distances [Thumb-Index, Thumb-Mid, Thumb-Pinky]

    Total length: exactly 144 float32 values with 0.0 padding for absent hands.
    """
    feature_vec = np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32)

    has_left = 1.0 if left_hand is not None else 0.0
    has_right = 1.0 if right_hand is not None else 0.0

    feature_vec[0] = has_left
    feature_vec[1] = has_right

    if left_hand is not None:
        # 63 normalized coordinates
        feature_vec[2:65] = left_hand.normalized_landmarks.astype(np.float32, copy=False).reshape(-1)
        # 5 finger curl ratios
        if left_hand.finger_curls is not None:
            feature_vec[65:70] = left_hand.finger_curls.astype(np.float32, copy=False)

    if right_hand is not None:
        # 63 normalized coordinates
        feature_vec[70:133] = right_hand.normalized_landmarks.astype(np.float32, copy=False).reshape(-1)
        # 5 finger curl ratios
        if right_hand.finger_curls is not None:
            feature_vec[133:138] = right_hand.finger_curls.astype(np.float32, copy=False)

    # Active hand kinematic descriptors (prioritize right hand, else left)
    active_hand = right_hand if right_hand is not None else left_hand
    if active_hand is not None:
        if active_hand.finger_splays is not None and len(active_hand.finger_splays) >= 4:
            # 3 inter-finger splays: Index-Mid, Mid-Ring, Ring-Pinky
            feature_vec[138:141] = active_hand.finger_splays[1:4]

        if active_hand.pinch_distances is not None and len(active_hand.pinch_distances) >= 4:
            # 3 key pinch distances: Thumb-Index, Thumb-Mid, Thumb-Pinky
            feature_vec[141] = active_hand.pinch_distances[0]  # T-I
            feature_vec[142] = active_hand.pinch_distances[1]  # T-M
            feature_vec[143] = active_hand.pinch_distances[3]  # T-P

    return feature_vec


@dataclass(slots=True)
class FrameTrackingResult:
    """Canonical frame-level output container from the hand tracking engine."""

    frame_index: int
    timestamp_ms: float
    inference_latency_ms: float
    left_hand: Optional[SingleHandData]
    right_hand: Optional[SingleHandData]
    presence_mask: Tuple[int, int]  # (has_left, has_right) as (0|1, 0|1)
    feature_vector: np.ndarray  # shape (144,), float32
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

