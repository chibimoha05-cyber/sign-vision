"""RTMPose SimCC hand tracking backend via OpenMMLab rtmlib and ONNXRuntime.

Provides a two-stage decoupled detection pipeline (RTMDet-Nano palm detector paired
with RTMPose-M keypoint estimator) that emits 21-landmark hand topologies matching
the MediaPipe schema, ensuring full compatibility with downstream kinematics and
feature vector contracts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

WRIST = 0
INDEX_MCP = 5
PINKY_MCP = 17


class RTMPoseBackend:
    """RTMPose SimCC hand tracking backend wrapper using rtmlib."""

    def __init__(
        self,
        mode: str = "lightweight",
        device: str = "cpu",
        min_score: float = 0.30,
    ) -> None:
        """Initializes RTMPose and RTMDet models via rtmlib.

        Args:
            mode: 'lightweight' (RTMDet-Nano + RTMPose-M) or 'balanced'.
            device: 'cpu' or 'cuda'.
            min_score: Minimum mean keypoint confidence threshold.
        """
        self.mode = mode
        self.device = device
        self.min_score = float(min_score)
        self.hand_model: Any = None
        self._is_initialized = False

        self._initialize()

    def _initialize(self) -> None:
        try:
            from rtmlib import Hand

            logger.info("Initializing RTMPose Hand model (mode=%s, device=%s)...", self.mode, self.device)
            self.hand_model = Hand(
                mode=self.mode,
                backend="onnxruntime",
                device=self.device,
            )
            self._is_initialized = True
            logger.info("RTMPose Hand backend initialized successfully.")
        except Exception as e:
            logger.warning("Failed to initialize RTMPose backend: %s", e)
            self.hand_model = None
            self._is_initialized = False

    @property
    def is_available(self) -> bool:
        return self._is_initialized and self.hand_model is not None

    @staticmethod
    def _estimate_handedness(kpts_norm: np.ndarray) -> str:
        """Estimates hand side using the palm vector cross product and screen position."""
        wrist = kpts_norm[WRIST, :2]
        index_mcp = kpts_norm[INDEX_MCP, :2]
        pinky_mcp = kpts_norm[PINKY_MCP, :2]

        v_idx = index_mcp - wrist
        v_pnk = pinky_mcp - wrist

        cross = float(v_idx[0] * v_pnk[1] - v_idx[1] * v_pnk[0])
        palm_center_x = float(wrist[0] + index_mcp[0] + pinky_mcp[0]) / 3.0

        if cross > 0.002:
            return "Right"
        elif cross < -0.002:
            return "Left"
        else:
            return "Left" if palm_center_x < 0.50 else "Right"

    def detect(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int = 0,
    ) -> List[Dict[str, Any]]:
        """Processes an image frame and returns detected hands with landmarks.

        Args:
            frame_bgr: Full BGR frame of shape (H, W, 3).
            timestamp_ms: Monotonic timestamp in milliseconds.

        Returns:
            List of dicts, each containing 'handedness', 'confidence',
            'landmarks' (21, 3), and 'world_landmarks' (21, 3).
        """
        if not self.is_available:
            return []

        h, w = frame_bgr.shape[:2]
        try:
            keypoints, scores = self.hand_model(frame_bgr)
        except Exception as e:
            logger.debug("RTMPose inference failed on frame: %s", e)
            return []

        if keypoints is None or len(keypoints) == 0:
            return []

        results: List[Dict[str, Any]] = []

        for hand_idx in range(len(keypoints)):
            kpts = keypoints[hand_idx]
            scrs = scores[hand_idx] if scores is not None and hand_idx < len(scores) else None

            mean_score = float(np.mean(scrs)) if scrs is not None else 0.50
            if mean_score < self.min_score:
                continue

            norm_landmarks = np.zeros((21, 3), dtype=np.float32)
            norm_landmarks[:, 0] = np.clip(kpts[:, 0] / max(w, 1), 0.0, 1.0)
            norm_landmarks[:, 1] = np.clip(kpts[:, 1] / max(h, 1), 0.0, 1.0)
            norm_landmarks[:, 2] = 0.0

            world_landmarks = np.zeros((21, 3), dtype=np.float32)
            world_landmarks[:, :2] = (norm_landmarks[:, :2] - norm_landmarks[0:1, :2]) * 0.20
            world_landmarks[:, 2] = 0.0

            handedness = self._estimate_handedness(norm_landmarks)

            results.append({
                "handedness": handedness,
                "confidence": mean_score,
                "landmarks": norm_landmarks,
                "world_landmarks": world_landmarks,
            })

        return results

    def close(self) -> None:
        """Releases underlying ONNX Runtime models."""
        self.hand_model = None
        self._is_initialized = False
