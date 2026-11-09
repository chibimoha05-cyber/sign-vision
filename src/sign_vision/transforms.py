"""Spatial Normalization and Adaptive Landmark Stabilization.

Mathematical Guarantees:
1. Translation Invariance: Wrist-origin centering shifts the coordinate space so P_0 = (0, 0, 0).
2. Scale Invariance: Normalizes by max 3D Euclidean distance from the wrist, bounding all points strictly to [-1.0, 1.0].
3. True One-Euro Filter (Casiez et al., CHI 2012): Timestamp-aware low-pass filter operating on
   raw 3D landmarks before projection. Stabilizes both screen rendering (pixels, bounding box)
   and normalized ML vectors without dynamic lag.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple
import numpy as np

# Safe epsilon to prevent division by zero in degenerate skeletons
EPSILON: float = 1e-6


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Applies wrist-relative translation and maximum-distance scale normalization.

    Given 21 3D points P_0, P_1, ..., P_20 where P_0 is the wrist:
    1. Relative translation: P'_i = P_i - P_0, so P'_0 = (0, 0, 0).
    2. Scale factor: S = max_{i in [1, 20]} ||P'_i||_2 (with safe epsilon floor).
    3. Normalized points: P_hat_i = P'_i / S.

    Args:
        landmarks: np.ndarray of shape (21, 3), float32/float64.

    Returns:
        np.ndarray of shape (21, 3), dtype float32, bounded in [-1.0, 1.0].
    """
    if landmarks.ndim != 2 or landmarks.shape[0] != 21 or landmarks.shape[1] != 3:
        raise ValueError(f"landmarks must have shape (21, 3), got {landmarks.shape}")

    coords = landmarks.astype(np.float64, copy=True)
    wrist = coords[0:1, :]  # Shape (1, 3)

    # 1. Translation Invariance: Centering wrist at origin (0, 0, 0)
    centered = coords - wrist  # wrist becomes [0.0, 0.0, 0.0]

    # 2. Scale Invariance: Calculate Euclidean distance of all points from wrist
    non_wrist = centered[1:, :]
    distances = np.linalg.norm(non_wrist, axis=1)

    max_dist = float(np.max(distances)) if len(distances) > 0 else 0.0
    scale = max(max_dist, EPSILON)

    normalized = centered / scale

    # Force exact 0.0 at wrist to eliminate any floating-point arithmetic residue
    normalized[0, :] = 0.0

    return normalized.astype(np.float32)


def compute_bounding_box(
    pixel_landmarks: np.ndarray,
    image_shape: Tuple[int, int],
    padding_ratio: float = 0.12,
) -> Tuple[int, int, int, int]:
    """Computes a padded (x_min, y_min, x_max, y_max) bounding box clamped to image bounds.

    Args:
        pixel_landmarks: (21, 2) int32 pixel coordinates.
        image_shape: (height, width) of the frame.
        padding_ratio: Fractional padding around the landmark extent.

    Returns:
        (x_min, y_min, x_max, y_max) integer pixel bounds.
    """
    h, w = image_shape[:2]
    x_min = int(np.min(pixel_landmarks[:, 0]))
    x_max = int(np.max(pixel_landmarks[:, 0]))
    y_min = int(np.min(pixel_landmarks[:, 1]))
    y_max = int(np.max(pixel_landmarks[:, 1]))

    box_w = max(x_max - x_min, 1)
    box_h = max(y_max - y_min, 1)
    pad_x = int(box_w * padding_ratio)
    pad_y = int(box_h * padding_ratio)

    clamped_x_min = max(0, x_min - pad_x)
    clamped_y_min = max(0, y_min - pad_y)
    clamped_x_max = min(w - 1, x_max + pad_x)
    clamped_y_max = min(h - 1, y_max + pad_y)

    return (clamped_x_min, clamped_y_min, clamped_x_max, clamped_y_max)


class OneEuroFilter:
    """Vectorized One-Euro Filter for 3D landmarks (Casiez et al., CHI 2012).

    Parameters:
        min_cutoff: Minimum cutoff frequency (Hz). Lower = stronger jitter suppression when still.
        beta: Speed coefficient. Higher = faster response / lower lag during rapid motions.
        d_cutoff: Cutoff frequency (Hz) for velocity derivation.
    """

    def __init__(
        self,
        min_cutoff: float = 1.8,
        beta: float = 0.15,
        d_cutoff: float = 1.2,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: Optional[np.ndarray] = None
        self.dx_prev: Optional[np.ndarray] = None
        self.t_prev: Optional[float] = None

    def reset(self) -> None:
        """Resets filter memory."""
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _alpha(dt: float, cutoff: np.ndarray | float) -> np.ndarray | float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: np.ndarray, timestamp_sec: float) -> np.ndarray:
        """Applies adaptive filtering on an input array of coordinates (e.g., 21x3).

        Args:
            x: Input array (float32 or float64).
            timestamp_sec: Monotonic timestamp in seconds.

        Returns:
            Smoothed array with identical shape, dtype float32.
        """
        current_x = x.astype(np.float64, copy=False)
        t = float(timestamp_sec)

        if self.t_prev is None or (t - self.t_prev) <= 0.0 or (t - self.t_prev) > 1.0:
            self.x_prev = current_x.copy()
            self.dx_prev = np.zeros_like(current_x)
            self.t_prev = t
            return x.astype(np.float32, copy=True)

        dt = t - self.t_prev

        # 1. Estimate and filter the velocity derivative
        dx = (current_x - self.x_prev) / dt
        alpha_d = self._alpha(dt, self.d_cutoff)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev

        # 2. Compute dynamic cutoff based on Euclidean speed per landmark
        speed = np.linalg.norm(dx_hat, axis=-1, keepdims=True)
        cutoff = self.min_cutoff + self.beta * speed
        alpha = self._alpha(dt, cutoff)

        # 3. Filter the signal with the dynamic cutoff
        x_hat = alpha * current_x + (1.0 - alpha) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat.astype(np.float32)


class HandStabilizer:
    """Production Multi-Hand Stabilizer supporting One-Euro and Adaptive EMA filtering.

    Provides smoothing at the source on raw (21, 3) landmarks so that:
    1. UI display skeleton (pixel_landmarks, bounding_box) is stabilized without jitter.
    2. Normalized feature vectors (normalized_landmarks) are rock-solid for ML inference.
    """

    def __init__(
        self,
        mode: Literal["one_euro", "fast", "unfiltered"] = "one_euro",
        min_cutoff: float = 1.8,
        beta: float = 0.15,
        d_cutoff: float = 1.2,
        max_missing_frames: int = 5,
        alpha_slow: float = 0.45,
        alpha_fast: float = 0.85,
        v_low: float = 0.015,
        v_high: float = 0.080,
    ) -> None:
        self.mode = mode
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.max_missing_frames = int(max_missing_frames)

        # Legacy EMA parameters (kept for backward compatibility)
        self.alpha_slow = float(alpha_slow)
        self.alpha_fast = float(alpha_fast)
        self.v_low = float(v_low)
        self.v_high = float(v_high)

        # Raw image-space filters per hand (strictly independent per joint)
        self._one_euro_filters: Dict[str, OneEuroFilter] = {
            "Left": self._make_filter(),
            "Right": self._make_filter(),
        }

        # World metric 3D filters per hand
        self._world_filters: Dict[str, OneEuroFilter] = {
            "Left": self._make_world_filter(),
            "Right": self._make_world_filter(),
        }

        # Legacy EMA history
        self._history: Dict[str, np.ndarray] = {}
        self._missing_count: Dict[str, int] = {"Left": 0, "Right": 0}

    def _make_filter(self) -> OneEuroFilter:
        if self.mode == "fast":
            return OneEuroFilter(min_cutoff=2.5, beta=0.10, d_cutoff=1.0)
        return OneEuroFilter(min_cutoff=self.min_cutoff, beta=self.beta, d_cutoff=self.d_cutoff)

    def _make_world_filter(self) -> OneEuroFilter:
        if self.mode == "fast":
            return OneEuroFilter(min_cutoff=2.5, beta=0.08, d_cutoff=1.0)
        return OneEuroFilter(min_cutoff=1.5, beta=0.04, d_cutoff=1.0)

    def set_mode(self, mode: Literal["one_euro", "fast", "unfiltered"]) -> None:
        """Dynamically updates the filtering preset mode."""
        self.mode = mode
        self.reset()

    def reset(self, handedness: Optional[Literal["Left", "Right"]] = None) -> None:
        """Resets the filter state for one or all hands."""
        if handedness is not None:
            self._one_euro_filters[handedness] = self._make_filter()
            self._world_filters[handedness] = self._make_world_filter()
            self._history.pop(handedness, None)
            self._missing_count[handedness] = 0
        else:
            self._one_euro_filters = {
                "Left": self._make_filter(),
                "Right": self._make_filter(),
            }
            self._world_filters = {
                "Left": self._make_world_filter(),
                "Right": self._make_world_filter(),
            }
            self._history.clear()
            self._missing_count = {"Left": 0, "Right": 0}

    def register_missing(self, handedness: Literal["Left", "Right"]) -> None:
        """Notifies the stabilizer of a missing frame, resetting state after max_missing_frames."""
        self._missing_count[handedness] = self._missing_count.get(handedness, 0) + 1
        if self._missing_count[handedness] >= self.max_missing_frames:
            self.reset(handedness)

    def filter_raw(
        self,
        raw_landmarks: np.ndarray,
        handedness: Literal["Left", "Right"],
        timestamp_sec: float,
    ) -> np.ndarray:
        """Smooths raw image-space landmarks (21, 3) at timestamp_sec with per-joint independence."""
        self._missing_count[handedness] = 0

        if self.mode == "unfiltered":
            return raw_landmarks.astype(np.float32, copy=True)

        filt = self._one_euro_filters.get(handedness)
        if filt is None:
            filt = self._make_filter()
            self._one_euro_filters[handedness] = filt

        return filt.filter(raw_landmarks, timestamp_sec)

    def filter_world(
        self,
        world_landmarks: np.ndarray,
        handedness: Literal["Left", "Right"],
        timestamp_sec: float,
    ) -> np.ndarray:
        """Smooths metric 3D Cartesian world landmarks (21, 3) independently per joint."""
        if self.mode == "unfiltered":
            return world_landmarks.astype(np.float32, copy=True)

        filt = self._world_filters.get(handedness)
        if filt is None:
            filt = self._make_world_filter()
            self._world_filters[handedness] = filt

        return filt.filter(world_landmarks, timestamp_sec)

    def filter(
        self,
        landmarks: np.ndarray,
        handedness: Literal["Left", "Right"],
        timestamp_sec: Optional[float] = None,
    ) -> np.ndarray:
        """Applies stabilization to normalized landmarks (backward compatible interface)."""
        self._missing_count[handedness] = 0
        current = landmarks.astype(np.float32, copy=True)

        if self.mode == "unfiltered":
            return current

        if timestamp_sec is not None:
            filt = self._one_euro_filters.get(handedness)
            if filt is None:
                filt = self._make_filter()
                self._one_euro_filters[handedness] = filt
            smoothed = filt.filter(current, timestamp_sec)
            smoothed[0, :] = 0.0  # Anchor wrist origin
            return smoothed

        # Fallback to adaptive EMA when no timestamp is passed
        prev = self._history.get(handedness)
        if prev is None:
            self._history[handedness] = current.copy()
            return current

        displacement = float(np.linalg.norm(current - prev) / np.sqrt(21.0))
        if displacement >= self.v_high:
            alpha = self.alpha_fast
        elif displacement <= self.v_low:
            alpha = self.alpha_slow
        else:
            ratio = (displacement - self.v_low) / (self.v_high - self.v_low)
            alpha = self.alpha_slow + ratio * (self.alpha_fast - self.alpha_slow)

        smoothed = (alpha * current) + ((1.0 - alpha) * prev)
        smoothed[0, :] = 0.0
        self._history[handedness] = smoothed.copy()
        return smoothed.astype(np.float32)
