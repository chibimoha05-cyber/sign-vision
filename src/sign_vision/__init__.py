"""sign_vision: Real-time biomechanical hand tracking and kinematic feature extraction.

Provides the public API for the sign language perception pipeline including the
tracker engine, landmark transforms, kinematics, gesture state machine, and
the canonical 144-dimensional feature vector contract.
"""

from sign_vision.schema import (
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
from sign_vision.capture import ThreadedCamera
from sign_vision.kinematics import (
    FINGER_NAMES,
    HandKinematicsEngine,
    KinematicFeatures,
)
from sign_vision.gestures import (
    GestureArbiter,
    GestureCandidate,
    GestureName,
    GestureState,
    HandGestureStateMachine,
)
from sign_vision.transforms import (
    EPSILON,
    HandStabilizer,
    OneEuroFilter,
    compute_bounding_box,
    normalize_landmarks,
)
from sign_vision.engine import SignHandTracker, SignHandTrackerOptions

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
