"""Sign Vision: Phase 1 Hand-Tracking & Feature Extraction Engine.

Ultra-low-latency real-time pipeline capturing frames, tracking 3D hand landmarks via MediaPipe,
applying wrist-centered translation and scale normalization with adaptive EMA smoothing,
and outputting strict, deterministic 128-float feature vectors for ML training and real-time inference.
"""

from sign_vision.contracts import (
    COORDINATES_PER_LANDMARK,
    FEATURE_DIM_V1,
    FEATURE_DIM_V2,
    HAND_FEATURE_DIM,
    LANDMARK_COUNT,
    TOTAL_FEATURE_DIM,
    FrameTrackingResult,
    SingleHandData,
    build_feature_vector,
    build_feature_vector_v1,
)
from sign_vision.camera import ThreadedCamera
from sign_vision.kinematics import (
    FINGER_NAMES,
    HandKinematicsEngine,
    KinematicFeatures,
)
from sign_vision.gesture_engine import (
    GestureArbiter,
    GestureCandidate,
    GestureName,
    GestureState,
    HandGestureStateMachine,
)
from sign_vision.normalizer import (
    EPSILON,
    HandStabilizer,
    OneEuroFilter,
    compute_bounding_box,
    normalize_landmarks,
)
from sign_vision.tracker import SignHandTracker, SignHandTrackerOptions

__version__ = "0.2.1"

__all__ = [
    "SignHandTracker",
    "SignHandTrackerOptions",
    "ThreadedCamera",
    "SingleHandData",
    "FrameTrackingResult",
    "HandKinematicsEngine",
    "KinematicFeatures",
    "FINGER_NAMES",
    "GestureName",
    "GestureState",
    "GestureCandidate",
    "GestureArbiter",
    "HandGestureStateMachine",
    "build_feature_vector",
    "build_feature_vector_v1",
    "normalize_landmarks",
    "HandStabilizer",
    "OneEuroFilter",
    "compute_bounding_box",
    "LANDMARK_COUNT",
    "COORDINATES_PER_LANDMARK",
    "HAND_FEATURE_DIM",
    "TOTAL_FEATURE_DIM",
    "FEATURE_DIM_V1",
    "FEATURE_DIM_V2",
    "EPSILON",
]

