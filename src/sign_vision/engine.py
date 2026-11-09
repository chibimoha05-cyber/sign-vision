"""Core Real-Time HandTracker Engine with Dual-Mode Recovery & Disambiguation.

Phase 3.0 Architecture:
1. Dual-Mode Instant Recovery: Seamlessly blends MediaPipe RunningMode.VIDEO
   with instantaneous RunningMode.IMAGE full-frame palm re-detection whenever
   inter-frame wrist velocity exceeds fast-motion thresholds or crops lag.
2. Inter-Hand Object Keypoint Similarity (OKS) & IoU Non-Maximum Suppression (NMS):
   Eliminates duplicate phantom skeletons when hands collide or cross.
3. Bipartite Hungarian Spatial & Anatomical Association:
   Optimally pairs detections to persistent Left/Right tracks using:
   - Kinematic trajectory & Kalman-predicted palm centers (60%)
   - Handedness classification agreement (25%)
   - Anatomical wrist-to-MCP orientation vector continuity (15%)
4. Linear Velocity Extrapolation Coasting (1 frame max, only at low velocity):
   Eliminates frozen skeleton hallucinations during rapid signing.
5. Multi-Backend Support: MediaPipe Tasks Hybrid <-> OpenMMLab RTMPose SimCC.
6. Pure Visual Detector Fidelity: Zero coordinate warping from rigidity projection.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Literal, Optional, Tuple
import urllib.request
import cv2
import numpy as np

from dataclasses import dataclass
from sign_vision.schema import (
    LANDMARK_COUNT,
    TOTAL_FEATURE_DIM,
    FEATURE_DIM_V1,
    FEATURE_DIM_V2,
    FrameTrackingResult,
    SingleHandData,
    build_feature_vector,
)
from sign_vision.kinematics import (
    FINGER_NAMES,
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_TIP,
    RING_MCP,
    RING_TIP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
    HandKinematicsEngine,
    KinematicFeatures,
)
from sign_vision.gestures import HandGestureStateMachine
from sign_vision.transforms import (
    HandStabilizer,
    OneEuroFilter,
    compute_bounding_box,
    normalize_landmarks,
)
from sign_vision.backends import RTMPoseBackend

logger = logging.getLogger(__name__)

HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# Standard Hand Skeleton Connections (21 landmarks)
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (13, 17), (17, 18), (18, 19), (19, 20),
    # Palm base closure
    (0, 17),
]


def ensure_model_asset(target_path: str = "models/hand_landmarker.task") -> Optional[str]:
    """Ensures that the official MediaPipe hand_landmarker.task model is present."""
    model_file = Path(target_path).resolve()
    if model_file.exists() and model_file.stat().st_size > 1_000_000:
        return str(model_file)

    model_file.parent.mkdir(parents=True, exist_ok=True)
    temp_target = model_file.with_suffix(".download.tmp")

    logger.info("Downloading official MediaPipe HandLandmarker model to %s...", model_file)
    try:
        urllib.request.urlretrieve(HAND_LANDMARKER_MODEL_URL, str(temp_target))
        if temp_target.stat().st_size > 1_000_000:
            temp_target.replace(model_file)
            logger.info("Successfully downloaded and verified %s (size: %d bytes)", model_file, model_file.stat().st_size)
            return str(model_file)
        if temp_target.exists():
            temp_target.unlink()
    except Exception as e:
        logger.warning("Could not download model asset: %s. Will attempt Solutions backend.", e)
        if temp_target.exists():
            temp_target.unlink()

    return None


@dataclass
class SignHandTrackerOptions:
    """Configuration options for SignHandTracker (Phase 3.0 High-Velocity)."""

    model_path: Optional[str] = "models/hand_landmarker.task"
    max_num_hands: int = 2
    min_detection_confidence: float = 0.40
    min_presence_confidence: float = 0.35
    min_tracking_confidence: float = 0.35
    enable_smoothing: bool = True
    smoothing_mode: Literal["one_euro", "fast", "unfiltered"] = "one_euro"
    inference_width: Optional[int] = 640  # Native 640x480 by default for sharp finger boundaries
    enhance_lighting: bool = False
    sharpen_finger_edges: bool = False  # Disabled by default to prevent webcam ISO chroma noise amplification
    static_image_mode_on_drift: bool = True  # Instantaneous re-alignment without spatial hysteresis when confidence drops
    max_coast_frames: int = 1  # 1 frame max to eliminate frozen skeletons during fast movement
    enforce_bone_rigidity: bool = False  # Disabled: never distort screen coordinates with synthetic bone lengths
    preferred_backend: Literal["auto", "tasks", "solutions", "mediapipe", "rtmpose"] = "auto"
    alpha_slow: float = 0.45
    alpha_fast: float = 0.85


class _HandTrackState:
    """Internal spatial identity tracker and temporal coasting buffer per hand."""

    def __init__(self, handedness: Literal["Left", "Right"]) -> None:
        self.handedness: Literal["Left", "Right"] = handedness
        self.is_locked: bool = False
        self.coast_count: int = 0
        self.last_wrist_norm: np.ndarray = np.array([0.5, 0.5], dtype=np.float32)
        self.last_palm_center: np.ndarray = np.array([0.5, 0.5], dtype=np.float32)
        self.last_raw_smoothed: Optional[np.ndarray] = None
        self.last_world_smoothed: Optional[np.ndarray] = None
        self.last_confidence: float = 0.0
        self.last_timestamp_sec: float = 0.0
        self.wrist_velocity: float = 0.0  # Normalized units per second
        self.palm_velocity_vec: np.ndarray = np.zeros(2, dtype=np.float32)
        self.joint_velocities: Optional[np.ndarray] = None  # (21, 3) units per second
        self.last_palm_orientation: Optional[np.ndarray] = None  # (2,) unit vector P9 - P0
        self.consecutive_flip_count: int = 0
        self.last_curls: Optional[np.ndarray] = None
        self.last_splays: Optional[np.ndarray] = None
        self.last_pinches: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.is_locked = False
        self.coast_count = 0
        self.last_raw_smoothed = None
        self.last_world_smoothed = None
        self.last_confidence = 0.0
        self.wrist_velocity = 0.0
        self.palm_velocity_vec = np.zeros(2, dtype=np.float32)
        self.joint_velocities = None
        self.last_palm_orientation = None
        self.consecutive_flip_count = 0
        self.last_curls = None
        self.last_splays = None
        self.last_pinches = None


class SignHandTracker:
    """Production-grade real-time hand tracker with Phase 3.0 High-Velocity Disambiguation."""

    def __init__(
        self,
        model_path: Optional[str] = "models/hand_landmarker.task",
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.40,
        min_presence_confidence: float = 0.35,
        min_tracking_confidence: float = 0.35,
        enable_smoothing: bool = True,
        smoothing_mode: Literal["one_euro", "fast", "unfiltered"] = "one_euro",
        inference_width: Optional[int] = 640,
        enhance_lighting: bool = False,
        sharpen_finger_edges: bool = False,
        static_image_mode_on_drift: bool = True,
        max_coast_frames: int = 1,
        enforce_bone_rigidity: bool = False,
        preferred_backend: Literal["auto", "tasks", "solutions", "mediapipe", "rtmpose"] = "auto",
        alpha_slow: float = 0.45,
        alpha_fast: float = 0.85,
        options: Optional[SignHandTrackerOptions] = None,
    ) -> None:
        """Initializes hand tracking engine with Phase 3.0 High-Velocity Precision."""
        if options is not None:
            self.model_path = options.model_path
            self.max_num_hands = options.max_num_hands
            self.min_detection_confidence = options.min_detection_confidence
            self.min_presence_confidence = options.min_presence_confidence
            self.min_tracking_confidence = options.min_tracking_confidence
            self.enable_smoothing = options.enable_smoothing
            smoothing_mode = options.smoothing_mode
            self.inference_width = options.inference_width
            self.enhance_lighting = options.enhance_lighting
            self.sharpen_finger_edges = options.sharpen_finger_edges
            self.static_image_mode_on_drift = options.static_image_mode_on_drift
            self.max_coast_frames = int(options.max_coast_frames)
            self.enforce_bone_rigidity = options.enforce_bone_rigidity
            preferred_backend = options.preferred_backend
            alpha_slow = options.alpha_slow
            alpha_fast = options.alpha_fast
        else:
            self.model_path = model_path
            self.max_num_hands = max_num_hands
            self.min_detection_confidence = min_detection_confidence
            self.min_presence_confidence = min_presence_confidence
            self.min_tracking_confidence = min_tracking_confidence
            self.enable_smoothing = enable_smoothing
            self.inference_width = inference_width
            self.enhance_lighting = enhance_lighting
            self.sharpen_finger_edges = sharpen_finger_edges
            self.static_image_mode_on_drift = static_image_mode_on_drift
            self.max_coast_frames = int(max_coast_frames)
            self.enforce_bone_rigidity = enforce_bone_rigidity

        self.stabilizer: Optional[HandStabilizer] = (
            HandStabilizer(
                mode=smoothing_mode,
                alpha_slow=alpha_slow,
                alpha_fast=alpha_fast,
            )
            if self.enable_smoothing
            else None
        )

        self.kinematics: HandKinematicsEngine = HandKinematicsEngine()
        self.gesture_state_machines: Dict[str, HandGestureStateMachine] = {
            "Left": HandGestureStateMachine(hand_label="Left", confirm_frames=2, release_frames=2),
            "Right": HandGestureStateMachine(hand_label="Right", confirm_frames=2, release_frames=2),
        }

        # Spatial trajectory and occlusion coasting states
        self._tracks: Dict[str, _HandTrackState] = {
            "Left": _HandTrackState("Left"),
            "Right": _HandTrackState("Right"),
        }

        self._frame_count: int = 0
        self._last_timestamp_ms: int = 0
        self.active_backend_name: str = "none"
        self.redetect_trigger_count: int = 0

        # Backend instances
        self._tasks_landmarker_video: Any = None
        self._tasks_landmarker_image: Any = None
        self._solutions_hands: Any = None
        self._rtmpose_backend: Optional[RTMPoseBackend] = None

        # Attempt to prepare RTMPose backend in background
        try:
            self._rtmpose_backend = RTMPoseBackend()
        except Exception as e:
            logger.debug("RTMPose backend could not be pre-initialized: %s", e)
            self._rtmpose_backend = None

        self._init_backend(preferred_backend)

    @property
    def _tasks_landmarker(self) -> Any:
        return self._tasks_landmarker_video

    def _init_backend(self, preferred: str) -> None:
        """Initializes vision backend with automatic fallback."""
        pref = preferred.lower()

        # 1. Explicit RTMPose preference
        if pref == "rtmpose":
            if self._rtmpose_backend is not None and self._rtmpose_backend.is_available:
                self.active_backend_name = "rtmpose"
                logger.info("Initialized RTMPose SimCC hand tracking backend successfully.")
                return
            else:
                logger.warning("RTMPose requested but unavailable. Falling back to MediaPipe.")
                pref = "auto"

        # 2. Explicit Solutions preference
        if pref == "solutions":
            try:
                import mediapipe as mp

                if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                    mp_hands = mp.solutions.hands
                    self._solutions_hands = mp_hands.Hands(
                        static_image_mode=False,
                        max_num_hands=self.max_num_hands,
                        min_detection_confidence=self.min_detection_confidence,
                        min_tracking_confidence=self.min_tracking_confidence,
                    )
                    self.active_backend_name = "solutions"
                    logger.info("Initialized MediaPipe Solutions Hands backend successfully.")
                    return
                else:
                    logger.info("MediaPipe legacy 'solutions' not present in installed version. Falling back to Tasks.")
                    pref = "tasks"
            except Exception as e:
                logger.warning("Solutions backend error: %s. Falling back to Tasks.", e)
                pref = "tasks"

        # 3. MediaPipe Tasks (VIDEO mode + IMAGE full-frame recovery fallback)
        if pref in ("auto", "tasks", "mediapipe"):
            try:
                resolved_model = ensure_model_asset(self.model_path or "models/hand_landmarker.task")
                if resolved_model is not None:
                    from mediapipe.tasks import python
                    from mediapipe.tasks.python import vision

                    base_options = python.BaseOptions(model_asset_path=resolved_model)

                    # Video landmarker for tracking continuity
                    options_video = vision.HandLandmarkerOptions(
                        base_options=base_options,
                        running_mode=vision.RunningMode.VIDEO,
                        num_hands=self.max_num_hands,
                        min_hand_detection_confidence=self.min_detection_confidence,
                        min_hand_presence_confidence=self.min_presence_confidence,
                        min_tracking_confidence=self.min_tracking_confidence,
                    )
                    self._tasks_landmarker_video = vision.HandLandmarker.create_from_options(options_video)

                    # Image landmarker for zero-lag full-frame palm re-detection
                    options_image = vision.HandLandmarkerOptions(
                        base_options=base_options,
                        running_mode=vision.RunningMode.IMAGE,
                        num_hands=self.max_num_hands,
                        min_hand_detection_confidence=self.min_detection_confidence,
                        min_hand_presence_confidence=self.min_presence_confidence,
                        min_tracking_confidence=self.min_tracking_confidence,
                    )
                    self._tasks_landmarker_image = vision.HandLandmarker.create_from_options(options_image)

                    self.active_backend_name = "tasks"
                    logger.info("Initialized MediaPipe Tasks (Dual VIDEO+IMAGE) backend successfully.")
                    return
            except Exception as e:
                logger.warning("MediaPipe Tasks backend initialization failed (%s). Falling back to RTMPose.", e)

        # 4. Fallback to RTMPose if available
        if self._rtmpose_backend is not None and self._rtmpose_backend.is_available:
            self.active_backend_name = "rtmpose"
            logger.info("Falling back to RTMPose backend.")
            return

        if self._tasks_landmarker_video is None and self._solutions_hands is None:
            raise RuntimeError("Failed to initialize any hand tracking backend (Tasks, Solutions, or RTMPose).")

    def switch_backend(self, backend: Optional[str] = None) -> str:
        """Dynamically switches tracking backend between MediaPipe Tasks Hybrid and RTMPose SimCC."""
        if backend is None:
            # Toggle between tasks and rtmpose
            if self.active_backend_name in ("tasks", "solutions", "mediapipe"):
                target = "rtmpose" if (self._rtmpose_backend and self._rtmpose_backend.is_available) else "tasks"
            else:
                target = "tasks"
        else:
            target = backend.lower()

        if target == "rtmpose":
            if self._rtmpose_backend is not None and self._rtmpose_backend.is_available:
                self.active_backend_name = "rtmpose"
                logger.info("Switched active backend to RTMPose SimCC.")
            else:
                logger.warning("RTMPose backend is not available. Remaining on %s.", self.active_backend_name)
        elif target in ("tasks", "mediapipe", "auto"):
            if self._tasks_landmarker_video is not None:
                self.active_backend_name = "tasks"
                logger.info("Switched active backend to MediaPipe Tasks Hybrid.")
            elif self._solutions_hands is not None:
                self.active_backend_name = "solutions"
                logger.info("Switched active backend to MediaPipe Solutions.")
            else:
                self._init_backend(preferred="tasks")

        return self.active_backend_name

    @staticmethod
    def _fast_enhance_lighting(bgr: np.ndarray) -> np.ndarray:
        """Fast 0.3ms contrast and luminance auto-stretch."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mean, std = cv2.meanStdDev(gray)
        alpha = float(np.clip(45.0 / max(float(std[0][0]), 1e-3), 0.85, 1.6))
        beta = float(128.0 - float(mean[0][0]) * alpha)
        return cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta)

    @staticmethod
    def _apply_edge_sharpening(img_bgr: np.ndarray) -> np.ndarray:
        """Adaptive unsharp masking (disabled by default to prevent webcam ISO chroma noise)."""
        gaussian = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=1.5)
        return cv2.addWeighted(img_bgr, 1.35, gaussian, -0.35, 0)

    @staticmethod
    def _apply_inter_hand_nms(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Suppresses phantom duplicate detections when two skeletons overlay the same physical hand."""
        if len(detections) <= 1:
            return detections

        # Sort detections by confidence descending
        sorted_dets = sorted(detections, key=lambda d: float(d["confidence"]), reverse=True)
        survivors: List[Dict[str, Any]] = []

        for det in sorted_dets:
            pts = det["landmarks"][:, :2]
            wrist = pts[0]
            is_duplicate = False

            for surv in survivors:
                s_pts = surv["landmarks"][:, :2]
                s_wrist = s_pts[0]

                # 1. Wrist distance
                d_wrist = float(np.linalg.norm(wrist - s_wrist))

                # 2. Mean per-joint Euclidean distance across all 21 landmarks
                d_pose = float(np.mean(np.linalg.norm(pts - s_pts, axis=1)))

                # 3. Bounding box IoU
                x1_min, y1_min = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
                x1_max, y1_max = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
                x2_min, y2_min = float(np.min(s_pts[:, 0])), float(np.min(s_pts[:, 1]))
                x2_max, y2_max = float(np.max(s_pts[:, 0])), float(np.max(s_pts[:, 1]))

                inter_w = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))
                inter_h = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
                inter_area = inter_w * inter_h
                area1 = max(1e-5, (x1_max - x1_min) * (y1_max - y1_min))
                area2 = max(1e-5, (x2_max - x2_min) * (y2_max - y2_min))
                iou = inter_area / (area1 + area2 - inter_area)

                # Duplicate suppression condition:
                # Virtually identical skeleton (d_pose < 0.045)
                # OR close wrists + close pose (d_wrist < 0.05 and d_pose < 0.075)
                # OR high IoU + close pose (iou > 0.70 and d_pose < 0.080)
                if (
                    d_pose < 0.045
                    or (d_wrist < 0.05 and d_pose < 0.075)
                    or (iou > 0.70 and d_pose < 0.080)
                ):
                    is_duplicate = True
                    break

            if not is_duplicate:
                survivors.append(det)

        return survivors

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: Optional[float] = None,
    ) -> FrameTrackingResult:
        """Processes a single BGR camera frame and returns canonical tracking results."""
        start_time = time.perf_counter()
        self._frame_count += 1
        full_h, full_w = frame_bgr.shape[:2]

        if timestamp_ms is None:
            now_ms = int(time.time() * 1000.0)
            if now_ms <= self._last_timestamp_ms:
                now_ms = self._last_timestamp_ms + 1
            cur_timestamp_ms = now_ms
        else:
            cur_timestamp_ms = int(timestamp_ms)
            if cur_timestamp_ms <= self._last_timestamp_ms:
                cur_timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = cur_timestamp_ms
        timestamp_sec = cur_timestamp_ms / 1000.0

        # 1. Inference Scaling
        if self.inference_width is not None and self.inference_width < full_w:
            inf_w = self.inference_width
            inf_h = int(round(full_h * (inf_w / full_w)))
            small_bgr = cv2.resize(frame_bgr, (inf_w, inf_h), interpolation=cv2.INTER_LINEAR)
        else:
            small_bgr = frame_bgr

        # 2. Optional Fast Lighting Normalization
        if self.enhance_lighting:
            small_bgr = self._fast_enhance_lighting(small_bgr)

        # 3. Optional Edge Contrast Sharpening (disabled by default)
        if self.sharpen_finger_edges:
            small_bgr = self._apply_edge_sharpening(small_bgr)

        # 4. Contiguous RGB layout for optimal C++ memory transfer
        rgb_frame = np.ascontiguousarray(cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB))
        raw_detections = self._run_detection(rgb_frame, cur_timestamp_ms, small_bgr)

        # 5. Spatial Hand Identity Association & Anti-Flip Hungarian Logic
        left_det, right_det = self._associate_and_stabilize_hands(raw_detections, timestamp_sec)

        # 6. Build SingleHandData with Source Smoothing, Kinematics & Velocity Extrapolation Coasting
        left_hand = self._build_hand_data(left_det, "Left", (full_h, full_w), timestamp_sec)
        right_hand = self._build_hand_data(right_det, "Right", (full_h, full_w), timestamp_sec)

        presence_mask = (1 if left_hand is not None else 0, 1 if right_hand is not None else 0)
        feature_vector = build_feature_vector(left_hand, right_hand)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        active_left = left_hand.confirmed_gesture if left_hand is not None else "none"
        active_right = right_hand.confirmed_gesture if right_hand is not None else "none"

        return FrameTrackingResult(
            frame_index=self._frame_count,
            timestamp_ms=float(cur_timestamp_ms),
            inference_latency_ms=float(latency_ms),
            left_hand=left_hand,
            right_hand=right_hand,
            presence_mask=presence_mask,
            feature_vector=feature_vector,
            active_left_gesture=active_left,
            active_right_gesture=active_right,
        )

    def _run_detection(
        self, rgb_frame: np.ndarray, timestamp_ms: int, small_bgr: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Executes active backend and extracts raw screen and metric world landmarks."""
        # 1. RTMPose Backend Execution
        if self.active_backend_name == "rtmpose" and self._rtmpose_backend is not None:
            results = self._rtmpose_backend.detect(small_bgr, timestamp_ms)
            return self._apply_inter_hand_nms(results)

        # 2. MediaPipe Tasks Backend Execution (with Dual-Mode Fast Recovery)
        if self.active_backend_name == "tasks" and self._tasks_landmarker_video is not None:
            import mediapipe as mp

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Check kinematic state of active tracks
            expected_hands = sum(1 for t in self._tracks.values() if t.is_locked and t.coast_count == 0)
            high_velocity = any(t.is_locked and t.wrist_velocity > 0.80 for t in self._tracks.values())

            # Primary execution: RunningMode.VIDEO (efficient frame-to-frame crop tracking)
            detection = self._tasks_landmarker_video.detect_for_video(mp_image, timestamp_ms)
            detected_hands = len(detection.hand_landmarks) if detection.hand_landmarks else 0

            # Evaluate confidence dip
            low_confidence = False
            if detection.handedness:
                for h_list in detection.handedness:
                    if h_list and h_list[0].score < 0.55:
                        low_confidence = True
                        break

            # Dual-Mode Trigger:
            # If high-speed motion occurs OR expected hands vanish OR confidence dips,
            # execute full-frame RunningMode.IMAGE palm detector fallback immediately!
            trigger_full_frame = (
                high_velocity
                or (expected_hands > 0 and detected_hands < expected_hands)
                or low_confidence
            )

            if trigger_full_frame and self._tasks_landmarker_image is not None:
                image_detection = self._tasks_landmarker_image.detect(mp_image)
                if image_detection.hand_landmarks and len(image_detection.hand_landmarks) >= detected_hands:
                    detection = image_detection
                    self.redetect_trigger_count += 1
                    logger.debug(
                        "Fast-motion full-frame palm detector triggered (expected=%d, video_det=%d, full_det=%d)",
                        expected_hands,
                        detected_hands,
                        len(image_detection.hand_landmarks),
                    )

            results: List[Dict[str, Any]] = []
            if detection.hand_landmarks:
                for idx, landmarks in enumerate(detection.hand_landmarks):
                    handedness_cat = (
                        detection.handedness[idx][0]
                        if idx < len(detection.handedness) and len(detection.handedness[idx]) > 0
                        else None
                    )
                    label = handedness_cat.category_name if handedness_cat else "Right"
                    score = float(handedness_cat.score) if handedness_cat else 0.9

                    # Screen landmarks
                    coords = np.empty((LANDMARK_COUNT, 3), dtype=np.float32)
                    for i, lm in enumerate(landmarks):
                        coords[i, 0] = lm.x
                        coords[i, 1] = lm.y
                        coords[i, 2] = lm.z

                    # Metric 3D world landmarks in meters
                    world_coords = np.zeros((LANDMARK_COUNT, 3), dtype=np.float32)
                    if detection.hand_world_landmarks and idx < len(detection.hand_world_landmarks):
                        for i, wlm in enumerate(detection.hand_world_landmarks[idx]):
                            world_coords[i, 0] = wlm.x
                            world_coords[i, 1] = wlm.y
                            world_coords[i, 2] = wlm.z
                    else:
                        world_coords = (coords - coords[0:1, :]) * 0.20

                    results.append({
                        "handedness": label,
                        "confidence": score,
                        "landmarks": coords,
                        "world_landmarks": world_coords,
                    })

            return self._apply_inter_hand_nms(results)

        # 3. MediaPipe Solutions Backend Execution
        if self.active_backend_name == "solutions" and self._solutions_hands is not None:
            detection = self._solutions_hands.process(rgb_frame)
            results = []
            if detection.multi_hand_landmarks:
                for idx, landmarks in enumerate(detection.multi_hand_landmarks):
                    handedness_class = (
                        detection.multi_handedness[idx].classification[0]
                        if detection.multi_handedness and idx < len(detection.multi_handedness)
                        else None
                    )
                    label = handedness_class.label if handedness_class else "Right"
                    score = float(handedness_class.score) if handedness_class else 0.9

                    coords = np.empty((LANDMARK_COUNT, 3), dtype=np.float32)
                    for i, lm in enumerate(landmarks.landmark):
                        coords[i, 0] = lm.x
                        coords[i, 1] = lm.y
                        coords[i, 2] = lm.z

                    world_coords = np.zeros((LANDMARK_COUNT, 3), dtype=np.float32)
                    if (
                        hasattr(detection, "multi_hand_world_landmarks")
                        and detection.multi_hand_world_landmarks
                        and idx < len(detection.multi_hand_world_landmarks)
                    ):
                        for i, wlm in enumerate(detection.multi_hand_world_landmarks[idx].landmark):
                            world_coords[i, 0] = wlm.x
                            world_coords[i, 1] = wlm.y
                            world_coords[i, 2] = wlm.z
                    else:
                        world_coords = (coords - coords[0:1, :]) * 0.20

                    results.append({
                        "handedness": label,
                        "confidence": score,
                        "landmarks": coords,
                        "world_landmarks": world_coords,
                    })

            return self._apply_inter_hand_nms(results)

        return []

    def _compute_track_cost(
        self,
        track: _HandTrackState,
        det: Dict[str, Any],
        timestamp_sec: float,
    ) -> float:
        """Computes matching cost between an active track and a candidate detection.

        Cost Components:
        - Kalman / Kinematic predicted palm center distance (w1 = 0.60)
        - Handedness classification agreement (w2 = 0.25)
        - Anatomical wrist-to-MCP orientation vector continuity (w3 = 0.15)
        """
        det_pts = det["landmarks"][:, :2]
        det_center = (det_pts[0] + det_pts[9]) / 2.0
        det_v_palm = det_pts[9] - det_pts[0]
        det_v_norm = det_v_palm / (np.linalg.norm(det_v_palm) + 1e-6)

        # 1. Kinematic center distance
        dt = max(timestamp_sec - track.last_timestamp_sec, 0.016)
        pred_center = track.last_palm_center + track.palm_velocity_vec * dt
        dist_center = float(np.linalg.norm(pred_center - det_center))
        w1_cost = min(dist_center / 0.40, 1.0)

        # 2. Handedness agreement
        det_hand = det["handedness"]
        w2_cost = 0.0 if det_hand == track.handedness else 1.0

        # 3. Palm orientation vector continuity
        if track.last_palm_orientation is not None:
            cos_sim = float(np.dot(track.last_palm_orientation, det_v_norm))
            w3_cost = float(np.clip((1.0 - cos_sim) / 2.0, 0.0, 1.0))
        else:
            w3_cost = 0.0

        return float(0.60 * w1_cost + 0.25 * w2_cost + 0.15 * w3_cost)

    def _associate_and_stabilize_hands(
        self,
        detections: List[Dict[str, Any]],
        timestamp_sec: float = 0.0,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Bipartite Hungarian / Optimal Assignment with Kinematic Trajectory & Anatomical Continuity."""
        if not detections:
            return None, None

        left_track = self._tracks["Left"]
        right_track = self._tracks["Right"]

        # Case 1: Initial state (neither track locked)
        if not left_track.is_locked and not right_track.is_locked:
            if len(detections) == 1:
                det = detections[0]
                return (det, None) if det["handedness"] == "Left" else (None, det)

            # Two detections without history: disambiguate duplicate labels by horizontal screen layout
            det0, det1 = detections[0], detections[1]
            if det0["handedness"] != det1["handedness"]:
                left = det0 if det0["handedness"] == "Left" else det1
                right = det1 if det1["handedness"] == "Right" else det0
                return left, right

            # Both claim same handedness: left-most is Left, right-most is Right
            c0_x = (det0["landmarks"][0, 0] + det0["landmarks"][9, 0]) / 2.0
            c1_x = (det1["landmarks"][0, 0] + det1["landmarks"][9, 0]) / 2.0
            if c0_x <= c1_x:
                det0["handedness"] = "Left"
                det1["handedness"] = "Right"
                return det0, det1
            else:
                det1["handedness"] = "Left"
                det0["handedness"] = "Right"
                return det1, det0

        # Case 2: Exactly one track is locked
        if left_track.is_locked != right_track.is_locked:
            locked_track = left_track if left_track.is_locked else right_track
            free_side = "Right" if locked_track.handedness == "Left" else "Left"

            if len(detections) == 1:
                det = detections[0]
                cost = self._compute_track_cost(locked_track, det, timestamp_sec)
                if cost < 0.65:
                    det["handedness"] = locked_track.handedness
                    return (det, None) if locked_track.handedness == "Left" else (None, det)
                else:
                    # Detection belongs to the other hand or locked track was dropped
                    return (det, None) if det["handedness"] == "Left" else (None, det)

            # Two detections with one locked track: assign closest to locked track, remainder to free
            det0, det1 = detections[0], detections[1]
            c0 = self._compute_track_cost(locked_track, det0, timestamp_sec)
            c1 = self._compute_track_cost(locked_track, det1, timestamp_sec)

            if c0 <= c1:
                det0["handedness"] = locked_track.handedness
                det1["handedness"] = free_side
                return (det0, det1) if locked_track.handedness == "Left" else (det1, det0)
            else:
                det1["handedness"] = locked_track.handedness
                det0["handedness"] = free_side
                return (det1, det0) if locked_track.handedness == "Left" else (det0, det1)

        # Case 3: Both tracks are locked (2-Hand Disambiguation & Crossing Resolution)
        if len(detections) == 1:
            det = detections[0]
            c_left = self._compute_track_cost(left_track, det, timestamp_sec)
            c_right = self._compute_track_cost(right_track, det, timestamp_sec)

            if c_left < c_right and c_left < 0.70:
                det["handedness"] = "Left"
                return det, None
            elif c_right <= c_left and c_right < 0.70:
                det["handedness"] = "Right"
                return None, det
            else:
                return (det, None) if det["handedness"] == "Left" else (None, det)

        # 2 detections and 2 locked tracks: solve optimal 2x2 bipartite assignment
        det0, det1 = detections[0], detections[1]
        c0_left = self._compute_track_cost(left_track, det0, timestamp_sec)
        c1_right = self._compute_track_cost(right_track, det1, timestamp_sec)
        cost_option_a = c0_left + c1_right  # det0 -> Left, det1 -> Right

        c1_left = self._compute_track_cost(left_track, det1, timestamp_sec)
        c0_right = self._compute_track_cost(right_track, det0, timestamp_sec)
        cost_option_b = c1_left + c0_right  # det1 -> Left, det0 -> Right

        if cost_option_a <= cost_option_b:
            det0["handedness"] = "Left"
            det1["handedness"] = "Right"
            return det0, det1
        else:
            det1["handedness"] = "Left"
            det0["handedness"] = "Right"
            return det1, det0

    def _build_hand_data(
        self,
        det: Optional[Dict[str, Any]],
        handedness: Literal["Left", "Right"],
        image_shape: Tuple[int, int],
        timestamp_sec: float,
    ) -> Optional[SingleHandData]:
        """Constructs SingleHandData with Source Smoothing, Kinematics & Velocity Extrapolation Coasting."""
        track = self._tracks[handedness]
        full_h, full_w = image_shape

        if det is not None:
            raw_coords = det["landmarks"].astype(np.float32, copy=True)
            raw_world = det.get("world_landmarks", np.zeros((LANDMARK_COUNT, 3), dtype=np.float32)).astype(np.float32, copy=True)
            conf = float(det["confidence"])

            # Static image mode on drift: clear hysteresis when confidence dips
            if self.static_image_mode_on_drift and conf < 0.40:
                track.consecutive_flip_count = 0

            # 1. Apply true One-Euro smoothing at source with independent per-joint speed
            if self.stabilizer is not None:
                smoothed_raw = self.stabilizer.filter_raw(raw_coords, handedness, timestamp_sec)
                smoothed_world = self.stabilizer.filter_world(raw_world, handedness, timestamp_sec)
            else:
                smoothed_raw = raw_coords
                smoothed_world = raw_world

            # 2. Extract Biomechanical Kinematic Features (pass world_landmarks for true 3D curls)
            _, kinematics = self.kinematics.extract_kinematics(
                smoothed_raw,
                handedness=handedness,
                enforce_rigidity=self.enforce_bone_rigidity,
                world_landmarks=smoothed_world,
            )

            # 3. Calculate Inter-Frame Velocities for Kinematic Trajectory Prediction
            dt = max(timestamp_sec - track.last_timestamp_sec, 0.001)
            if track.is_locked and track.last_raw_smoothed is not None:
                wrist_diff = smoothed_raw[0, :2] - track.last_raw_smoothed[0, :2]
                track.wrist_velocity = float(np.linalg.norm(wrist_diff) / dt)
                track.palm_velocity_vec = (wrist_diff / dt).astype(np.float32)
                track.joint_velocities = ((smoothed_raw - track.last_raw_smoothed) / dt).astype(np.float32)
            else:
                track.wrist_velocity = 0.0
                track.palm_velocity_vec = np.zeros(2, dtype=np.float32)
                track.joint_velocities = np.zeros_like(smoothed_raw)

            # Update track state
            track.is_locked = True
            track.coast_count = 0
            track.last_raw_smoothed = smoothed_raw.copy()
            track.last_world_smoothed = smoothed_world.copy()
            track.last_wrist_norm = smoothed_raw[0, :2].copy()
            track.last_palm_center = (smoothed_raw[0, :2] + smoothed_raw[9, :2]) / 2.0
            v_palm = smoothed_raw[9, :2] - smoothed_raw[0, :2]
            track.last_palm_orientation = (v_palm / (np.linalg.norm(v_palm) + 1e-6)).astype(np.float32)
            track.last_confidence = conf
            track.last_timestamp_sec = timestamp_sec
            track.last_curls = kinematics.finger_curls.copy()
            track.last_splays = kinematics.finger_splays.copy()
            track.last_pinches = kinematics.pinch_distances.copy()

            # 4. Pixel landmarks mapped to full display frame
            pixel_coords = np.empty((LANDMARK_COUNT, 2), dtype=np.int32)
            pixel_coords[:, 0] = np.clip(np.round(smoothed_raw[:, 0] * full_w), 0, full_w - 1).astype(np.int32)
            pixel_coords[:, 1] = np.clip(np.round(smoothed_raw[:, 1] * full_h), 0, full_h - 1).astype(np.int32)

            # 5. Spatial wrist-centering + scale invariance from smoothed coordinates
            norm_coords = normalize_landmarks(smoothed_raw)

            # 6. Bounding box calculation
            bbox = compute_bounding_box(pixel_coords, image_shape)

            # 7. Temporal Gesture State Machine
            confirmed_g, g_state = self.gesture_state_machines[handedness].process(
                binary_mask=kinematics.binary_mask,
                finger_curls=kinematics.finger_curls,
                pinch_distances=kinematics.pinch_distances,
                tilt_angle_deg=kinematics.tilt_angle_deg,
            )

            return SingleHandData(
                handedness=handedness,
                confidence=conf,
                raw_landmarks=smoothed_raw,
                pixel_landmarks=pixel_coords,
                normalized_landmarks=norm_coords,
                bounding_box=bbox,
                world_landmarks=smoothed_world,
                finger_curls=kinematics.finger_curls,
                finger_splays=kinematics.finger_splays,
                pinch_distances=kinematics.pinch_distances,
                is_coasting=False,
                binary_mask=kinematics.binary_mask,
                tilt_angle_deg=kinematics.tilt_angle_deg,
                confirmed_gesture=confirmed_g.value,
                gesture_state=g_state,
            )

        # Detection is absent: Velocity-Extrapolated Temporal Coasting
        if track.is_locked and track.last_raw_smoothed is not None:
            # Fatal Bug 5 Fix: Never freeze a static skeleton in mid-air during fast motion!
            # Only coast if wrist velocity is low (<= 0.40 norm-units/s) and coast_count < max_coast_frames
            if track.wrist_velocity <= 0.40 and track.coast_count < self.max_coast_frames:
                track.coast_count += 1
                dt = max(timestamp_sec - track.last_timestamp_sec, 0.016)

                # Linear velocity extrapolation: P_t = clip(P_{t-1} + 0.5 * V_{t-1} * dt, 0.0, 1.0)
                if track.joint_velocities is not None:
                    coasting_raw = np.clip(
                        track.last_raw_smoothed + 0.5 * track.joint_velocities * dt,
                        0.0,
                        1.0,
                    ).astype(np.float32)
                else:
                    coasting_raw = track.last_raw_smoothed.copy()

                coasting_world = (
                    track.last_world_smoothed.copy()
                    if track.last_world_smoothed is not None
                    else np.zeros((LANDMARK_COUNT, 3), dtype=np.float32)
                )
                decayed_conf = float(track.last_confidence * 0.70)

                pixel_coords = np.empty((LANDMARK_COUNT, 2), dtype=np.int32)
                pixel_coords[:, 0] = np.clip(np.round(coasting_raw[:, 0] * full_w), 0, full_w - 1).astype(np.int32)
                pixel_coords[:, 1] = np.clip(np.round(coasting_raw[:, 1] * full_h), 0, full_h - 1).astype(np.int32)

                norm_coords = normalize_landmarks(coasting_raw)
                bbox = compute_bounding_box(pixel_coords, image_shape)

                curls = track.last_curls if track.last_curls is not None else np.zeros(5, dtype=np.float32)
                splays = track.last_splays if track.last_splays is not None else np.zeros(4, dtype=np.float32)
                pinches = track.last_pinches if track.last_pinches is not None else np.zeros(4, dtype=np.float32)

                active_g = self.gesture_state_machines[handedness].active_gesture.value
                g_state = self.gesture_state_machines[handedness].state.value

                return SingleHandData(
                    handedness=handedness,
                    confidence=decayed_conf,
                    raw_landmarks=coasting_raw,
                    pixel_landmarks=pixel_coords,
                    normalized_landmarks=norm_coords,
                    bounding_box=bbox,
                    world_landmarks=coasting_world,
                    finger_curls=curls,
                    finger_splays=splays,
                    pinch_distances=pinches,
                    is_coasting=True,
                    binary_mask=0,
                    tilt_angle_deg=0.0,
                    confirmed_gesture=active_g,
                    gesture_state=g_state,
                )

            # Exceeded max coasting frames or fast-motion cutoff: hand has departed or dropped
            track.reset()
            self.gesture_state_machines[handedness].reset()
            if self.stabilizer is not None:
                self.stabilizer.register_missing(handedness)

        return None

    def draw_overlays(
        self,
        frame_bgr: np.ndarray,
        result: FrameTrackingResult,
        draw_hud: bool = True,
        camera_fps: float = 0.0,
        draw_kinematics: bool = True,
    ) -> np.ndarray:
        """Renders skeletal landmarks, bounding boxes, divergence rays, and biomechanical dashboard."""
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]

        colors = {
            "Left": {"bone": (255, 210, 0), "joint": (255, 255, 100), "wrist": (200, 255, 0), "text": (255, 220, 50)},
            "Right": {"bone": (40, 140, 255), "joint": (100, 180, 255), "wrist": (0, 120, 255), "text": (60, 160, 255)},
            "Coasting": {"bone": (0, 220, 255), "joint": (100, 240, 255), "wrist": (0, 200, 255), "text": (0, 230, 255)},
        }

        for hand_data in (result.left_hand, result.right_hand):
            if hand_data is None:
                continue

            pal = colors["Coasting"] if hand_data.is_coasting else colors[hand_data.handedness]
            pts = hand_data.pixel_landmarks

            # 1. Draw skeletal bones
            line_thickness = 1 if hand_data.is_coasting else 2
            for start_idx, end_idx in HAND_CONNECTIONS:
                p1 = (int(pts[start_idx, 0]), int(pts[start_idx, 1]))
                p2 = (int(pts[end_idx, 0]), int(pts[end_idx, 1]))
                cv2.line(annotated, p1, p2, pal["bone"], line_thickness, cv2.LINE_AA)

            # 2. Draw inter-finger divergence rays
            if draw_kinematics:
                mcp_indices = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
                tip_indices = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
                for m_idx, t_idx in zip(mcp_indices, tip_indices):
                    p_mcp = (int(pts[m_idx, 0]), int(pts[m_idx, 1]))
                    p_tip = (int(pts[t_idx, 0]), int(pts[t_idx, 1]))
                    # Subtle ray extension past fingertip
                    ray_dx = int((p_tip[0] - p_mcp[0]) * 0.35)
                    ray_dy = int((p_tip[1] - p_mcp[1]) * 0.35)
                    p_ext = (p_tip[0] + ray_dx, p_tip[1] + ray_dy)
                    cv2.line(annotated, p_tip, p_ext, (0, 255, 200), 1, cv2.LINE_AA)
                    cv2.circle(annotated, p_ext, 2, (0, 255, 200), -1, cv2.LINE_AA)

            # 3. Draw landmark joints
            for i in range(LANDMARK_COUNT):
                pt = (int(pts[i, 0]), int(pts[i, 1]))
                radius = 5 if i == 0 else 3
                c = pal["wrist"] if i == 0 else pal["joint"]
                cv2.circle(annotated, pt, radius, c, -1, cv2.LINE_AA)
                cv2.circle(annotated, pt, radius + 1, (20, 20, 20), 1, cv2.LINE_AA)

            # 4. Draw bounding box and label
            x_min, y_min, x_max, y_max = hand_data.bounding_box
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), pal["bone"], 1, cv2.LINE_AA)

            accent_len = max(8, min(20, (x_max - x_min) // 5))
            for cx, cy in [(x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)]:
                dx = accent_len if cx == x_min else -accent_len
                dy = accent_len if cy == y_min else -accent_len
                cv2.line(annotated, (cx, cy), (cx + dx, cy), pal["text"], 2, cv2.LINE_AA)
                cv2.line(annotated, (cx, cy), (cx, cy + dy), pal["text"], 2, cv2.LINE_AA)

            state_tag = "[COASTING]" if hand_data.is_coasting else f"{int(hand_data.confidence * 100)}%"
            label_text = f"{hand_data.handedness} {state_tag}"
            label_y = max(20, y_min - 6)
            cv2.putText(annotated, label_text, (x_min, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (10, 10, 10), 3, cv2.LINE_AA)
            cv2.putText(annotated, label_text, (x_min, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, pal["text"], 1, cv2.LINE_AA)

        # 5. Diagnostic HUD Top Banner
        if draw_hud:
            hud_bg_w, hud_bg_h = 350, 88
            overlay = annotated.copy()
            cv2.rectangle(overlay, (12, 12), (12 + hud_bg_w, 12 + hud_bg_h), (15, 18, 22), -1)
            cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)
            cv2.rectangle(annotated, (12, 12), (12 + hud_bg_w, 12 + hud_bg_h), (60, 70, 85), 1, cv2.LINE_AA)

            cam_str = f" | Cam: {camera_fps:4.1f} FPS" if camera_fps > 0 else ""
            txt_lat = f"Phase 3.0 | {result.inference_latency_ms:4.1f}ms{cam_str}"
            cv2.putText(annotated, txt_lat, (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 240, 200), 1, cv2.LINE_AA)

            l_tag = "L:COASTING" if (result.left_hand and result.left_hand.is_coasting) else ("L:LOCKED" if result.left_hand else "L:--")
            r_tag = "R:COASTING" if (result.right_hand and result.right_hand.is_coasting) else ("R:LOCKED" if result.right_hand else "R:--")
            l_col = (0, 220, 255) if (result.left_hand and result.left_hand.is_coasting) else ((255, 210, 0) if result.left_hand else (120, 130, 140))
            r_col = (0, 220, 255) if (result.right_hand and result.right_hand.is_coasting) else ((40, 140, 255) if result.right_hand else (120, 130, 140))

            cv2.putText(annotated, l_tag, (22, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.44, l_col, 1, cv2.LINE_AA)
            cv2.putText(annotated, r_tag, (160, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.44, r_col, 1, cv2.LINE_AA)

            filter_name = self.stabilizer.mode.upper() if self.stabilizer else "OFF"
            backend_label = "RTMPose" if self.active_backend_name == "rtmpose" else "MP-Hybrid"
            txt_status = f"Eng:{backend_label} | Filt:{filter_name} | Trig:{self.redetect_trigger_count}"
            cv2.putText(annotated, txt_status, (22, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 190, 200), 1, cv2.LINE_AA)

        # 6. Biomechanical Finger State Dashboard
        if draw_kinematics and result.has_any_hand:
            active_hand = result.right_hand if result.right_hand is not None else result.left_hand
            if active_hand is not None and active_hand.finger_curls is not None:
                panel_w, panel_h = 320, 135
                px = max(12, w - panel_w - 12)
                py = max(12, h - panel_h - 12)

                overlay = annotated.copy()
                cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (12, 15, 20), -1)
                cv2.addWeighted(overlay, 0.88, annotated, 0.12, 0, annotated)
                cv2.rectangle(annotated, (px, py), (px + panel_w, py + panel_h), (0, 200, 160), 1, cv2.LINE_AA)

                g_name = active_hand.confirmed_gesture.upper()
                g_st = active_hand.gesture_state
                hdr_title = f"{active_hand.handedness.upper()}: {g_name} [{g_st}] | M:{active_hand.binary_mask:02d} | T:{active_hand.tilt_angle_deg:+.0f}"
                cv2.putText(annotated, hdr_title, (px + 10, py + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 240, 200), 1, cv2.LINE_AA)

                for f_idx, name in enumerate(FINGER_NAMES):
                    curl = float(active_hand.finger_curls[f_idx])
                    if curl >= 0.65:
                        state_str = "EXTENDED"
                        state_col = (50, 220, 100)  # Green
                    elif curl <= 0.35:
                        state_str = "CURLED"
                        state_col = (40, 120, 255)  # Orange/Coral
                    else:
                        state_str = "PARTIAL"
                        state_col = (0, 220, 255)  # Yellow

                    # Visual progress bar
                    bar_w = 70
                    bar_filled = int(round(curl * bar_w))
                    bx = px + 80
                    by = py + 30 + f_idx * 20

                    cv2.putText(annotated, f"{name[:3].upper()}:", (px + 10, by + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 210, 220), 1, cv2.LINE_AA)
                    cv2.rectangle(annotated, (bx, by + 2), (bx + bar_w, by + 12), (50, 60, 70), -1)
                    if bar_filled > 0:
                        cv2.rectangle(annotated, (bx, by + 2), (bx + bar_filled, by + 12), state_col, -1)
                    cv2.rectangle(annotated, (bx, by + 2), (bx + bar_w, by + 12), (100, 110, 120), 1)

                    desc = f"{int(curl * 100):3d}% ({state_str})"
                    cv2.putText(annotated, desc, (bx + bar_w + 8, by + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, state_col, 1, cv2.LINE_AA)

        return annotated

    def close(self) -> None:
        """Releases underlying vision backends and resources."""
        if self._tasks_landmarker_video is not None:
            try:
                self._tasks_landmarker_video.close()
            except Exception:
                pass
            self._tasks_landmarker_video = None

        if self._tasks_landmarker_image is not None:
            try:
                self._tasks_landmarker_image.close()
            except Exception:
                pass
            self._tasks_landmarker_image = None

        if self._solutions_hands is not None:
            try:
                self._solutions_hands.close()
            except Exception:
                pass
            self._solutions_hands = None

        if self._rtmpose_backend is not None:
            try:
                self._rtmpose_backend.close()
            except Exception:
                pass
            self._rtmpose_backend = None

        if self.stabilizer is not None:
            self.stabilizer.reset()
        if hasattr(self, "kinematics"):
            self.kinematics.reset()
        for track in self._tracks.values():
            track.reset()
        for gsm in self.gesture_state_machines.values():
            gsm.reset()

    def __enter__(self) -> SignHandTracker:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
