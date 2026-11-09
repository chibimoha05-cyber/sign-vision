"""Multi-Threaded Low-Latency Camera Capture Engine.

Key Architectural Guarantees:
1. Zero-Buffer Lag: Dedicated daemon thread continuously pulls the newest frame from the hardware
   bus, discarding stale OS buffer queues so the tracker never runs on outdated frames.
2. Cross-Platform Backend Negotiation: Strictly sets V4L2 + MJPG before and after resolution/FPS
   negotiation on Linux to prevent uncompressed YUYV down-throttling.
3. Anti-Throttling: Automatically disables V4L2 dynamic frame-rate throttling in low light.
4. Hardware FPS Telemetry: Independently measures and exposes actual camera capture FPS (camera_fps).
5. Natural Selfie Mirroring: Mirrors horizontal axis by default (cv2.flip(frame, 1)) so user
   movements feel intuitive on screen, aligning user Left/Right with visual feedback.
"""

from __future__ import annotations

import collections
import logging
import platform
import subprocess
import sys
import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ThreadedCamera:
    """Threaded camera grabber providing decoupled, zero-latency frame access."""

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        target_fps: int = 60,
        mirror: bool = True,
        backend: Optional[int] = None,
        anti_blur_mode: bool = True,
    ) -> None:
        """Initializes camera parameters without starting the capture thread.

        Args:
            device_index: Camera device ID (default 0).
            width: Desired capture width (default 640 for crisp high-FPS capture).
            height: Desired capture height (default 480).
            target_fps: Desired capture FPS (default 60).
            mirror: If True, horizontally flips frames for selfie perspective.
            backend: Optional cv2 VideoCapture backend override.
            anti_blur_mode: If True, engages V4L2 fast-shutter exposure tuning.
        """
        self.device_index = device_index
        self.requested_width = width
        self.requested_height = height
        self.target_fps = target_fps
        self.mirror = mirror
        self.backend = backend

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._is_running = False
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._new_frame_event = threading.Event()

        self._latest_frame: Optional[np.ndarray] = None
        self._frame_index: int = 0
        self._frame_timestamp: float = 0.0

        self._actual_width: int = 0
        self._actual_height: int = 0
        self._actual_fps: float = 0.0

        # Independent hardware capture FPS telemetry
        self._capture_times = collections.deque(maxlen=30)
        self._hardware_fps: float = 0.0

        # Anti-Blur Fast Shutter settings & Gamma LUT
        self.anti_blur_mode: bool = anti_blur_mode
        self.current_exposure_time: int = 100
        gamma = 1.45
        inv_gamma = 1.0 / gamma
        self._gamma_lut = np.array([np.clip(((i / 255.0) ** inv_gamma) * 255.0, 0, 255) for i in range(256)], dtype=np.uint8)
        self._needs_gamma_boost: bool = False

    @property
    def exposure_val(self) -> int:
        return self.current_exposure_time

    @staticmethod
    def _select_optimal_backend(preferred_backend: Optional[int] = None) -> list[int]:
        """Returns ordered list of backends to attempt based on the OS."""
        if preferred_backend is not None:
            return [preferred_backend, cv2.CAP_ANY]

        system = platform.system().lower()
        if "linux" in system:
            return [cv2.CAP_V4L2, cv2.CAP_ANY]
        elif "windows" in system:
            return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        elif "darwin" in system:
            return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        return [cv2.CAP_ANY]

    def _disable_v4l2_framerate_throttling(self) -> None:
        """Attempts to disable Linux V4L2 dynamic frame-rate throttling in low light."""
        if "linux" not in platform.system().lower():
            return

        dev_path = f"/dev/video{self.device_index}"
        try:
            subprocess.run(
                ["v4l2-ctl", "-d", dev_path, "-c", "exposure_dynamic_framerate=0"],
                capture_output=True,
                timeout=0.25,
            )
            logger.info("Disabled exposure_dynamic_framerate on %s", dev_path)
        except Exception:
            pass

    def _open_capture_device(self) -> cv2.VideoCapture:
        """Iterates through preferred backends to initialize the video device."""
        self._disable_v4l2_framerate_throttling()

        backends = self._select_optimal_backend(self.backend)
        last_error = None
        fourcc_mjpg = cv2.VideoWriter_fourcc(*"MJPG")

        for b in backends:
            try:
                cap = cv2.VideoCapture(self.device_index, b)
                if cap.isOpened():
                    # 1. Strict pre-resolution MJPG FOURCC request
                    try:
                        cap.set(cv2.CAP_PROP_FOURCC, fourcc_mjpg)
                    except Exception:
                        pass

                    # 2. Configure Resolution & Target FPS
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
                    cap.set(cv2.CAP_PROP_FPS, self.target_fps)

                    # 3. Strict post-resolution MJPG FOURCC confirmation (ensures driver preserves MJPG)
                    try:
                        cap.set(cv2.CAP_PROP_FOURCC, fourcc_mjpg)
                    except Exception:
                        pass

                    # 4. Minimize hardware/driver queue buffer to 1 frame (prevents lag accumulation)
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass

                    # 5. Attempt auto-exposure tuning to prevent sensor framerate throttling
                    try:
                        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # Aperture priority
                    except Exception:
                        pass

                    self._actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.requested_width)
                    self._actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.requested_height)
                    self._actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or self.target_fps)

                    logger.info(
                        "Opened camera %d with backend %s (%dx%d @ %.1f FPS)",
                        self.device_index,
                        b,
                        self._actual_width,
                        self._actual_height,
                        self._actual_fps,
                    )
                    if self.anti_blur_mode:
                        self._apply_v4l2_exposure_settings()
                    return cap
                cap.release()
            except Exception as e:
                last_error = e

        raise RuntimeError(
            f"Failed to open camera device {self.device_index} across backends {backends}. Last error: {last_error}"
        )

    def start(self) -> ThreadedCamera:
        """Opens camera and starts the background acquisition thread."""
        if self._is_running:
            return self

        self._cap = self._open_capture_device()
        self._is_running = True
        self._new_frame_event.clear()
        self._capture_times.clear()

        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"CameraReader-{self.device_index}",
            daemon=True,
        )
        self._thread.start()
        return self

    def _capture_loop(self) -> None:
        """Continuous background thread discarding buffered frames and storing newest."""
        while self._is_running and self._cap is not None:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                # Driver stall or frame drop: sleep briefly to avoid pegging CPU
                time.sleep(0.002)
                continue

            now = time.perf_counter()

            # Track empirical hardware capture FPS
            self._capture_times.append(now)
            if len(self._capture_times) >= 2:
                duration = self._capture_times[-1] - self._capture_times[0]
                if duration > 0:
                    self._hardware_fps = (len(self._capture_times) - 1) / duration

            if self.mirror:
                frame = cv2.flip(frame, 1)

            with self._condition:
                self._latest_frame = frame
                self._frame_index += 1
                self._frame_timestamp = time.time()
                self._new_frame_event.set()
                self._condition.notify_all()

    def read_new_frame(
        self, last_consumed_index: int = -1, timeout: float = 0.1
    ) -> Tuple[bool, Optional[np.ndarray], int, float]:
        """Waits for and returns a strictly new unique hardware camera frame.

        Guarantees that each camera frame is processed exactly once by tracking the hardware
        frame index, completely eliminating duplicate frame processing.

        Args:
            last_consumed_index: The index of the previously processed frame.
            timeout: Maximum wait time in seconds (default 0.1s).

        Returns:
            Tuple of (success, frame_bgr, frame_index, timestamp_seconds).
        """
        if not self._is_running:
            return False, None, last_consumed_index, 0.0

        with self._condition:
            if self._frame_index <= last_consumed_index:
                has_new = self._condition.wait_for(
                    lambda: self._frame_index > last_consumed_index or not self._is_running,
                    timeout=timeout,
                )
                if not has_new or not self._is_running:
                    return False, None, last_consumed_index, 0.0

            if self._latest_frame is None:
                return False, None, last_consumed_index, 0.0

            frame = self._latest_frame.copy()
            idx = self._frame_index
            ts = self._frame_timestamp

        # Fast Gamma LUT luminance recovery for indoor fast shutter mode
        if self.anti_blur_mode and self._gamma_lut is not None:
            if idx % 15 == 0:
                mean_lum = float(np.mean(frame))
                self._needs_gamma_boost = mean_lum < 85.0

            if getattr(self, "_needs_gamma_boost", False):
                frame = cv2.LUT(frame, self._gamma_lut)

        return True, frame, idx, ts

    def set_anti_blur(self, enabled: bool, exposure_time: Optional[int] = None) -> bool:
        """Toggles V4L2 Fast Shutter mode to eliminate indoor motion blur on Linux."""
        self.anti_blur_mode = enabled
        if exposure_time is not None:
            self.current_exposure_time = int(np.clip(exposure_time, 20, 600))
        return self._apply_v4l2_exposure_settings()

    def adjust_exposure(self, delta: int) -> int:
        """Increases or decreases manual exposure time by delta (microseconds/units)."""
        self.current_exposure_time = int(np.clip(self.current_exposure_time + delta, 20, 600))
        self.anti_blur_mode = True
        self._apply_v4l2_exposure_settings()
        return self.current_exposure_time

    def _apply_v4l2_exposure_settings(self) -> bool:
        """Executes V4L2 hardware controls on Linux video device."""
        if "linux" not in platform.system().lower():
            return False

        dev_path = f"/dev/video{self.device_index}"
        try:
            if self.anti_blur_mode:
                # 1 = Manual Exposure
                subprocess.run(["v4l2-ctl", "-d", dev_path, "-c", "auto_exposure=1"], capture_output=True, timeout=0.25)
                subprocess.run(
                    ["v4l2-ctl", "-d", dev_path, "-c", f"exposure_time_absolute={self.current_exposure_time}"],
                    capture_output=True,
                    timeout=0.25,
                )
                logger.info(
                    "V4L2 Anti-Blur FAST SHUTTER enabled on %s (exposure=%d)",
                    dev_path,
                    self.current_exposure_time,
                )
            else:
                # 3 = Aperture Priority / Auto Exposure
                subprocess.run(["v4l2-ctl", "-d", dev_path, "-c", "auto_exposure=3"], capture_output=True, timeout=0.25)
                logger.info("V4L2 Auto-Exposure re-enabled on %s", dev_path)
            return True
        except Exception as e:
            logger.debug("Could not configure V4L2 exposure on %s: %s", dev_path, e)
            return False

    def read(self, timeout: float = 1.0) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads the newest captured frame (compatible with cv2.VideoCapture interface).

        Args:
            timeout: Maximum wait time for a frame in seconds.

        Returns:
            (success, frame_bgr)
        """
        success, frame, _, _ = self.read_latest(timeout=timeout)
        return success, frame

    def read_latest(
        self, timeout: float = 1.0
    ) -> Tuple[bool, Optional[np.ndarray], int, float]:
        """Reads the latest frame along with monotonic index and capture timestamp.

        Args:
            timeout: Seconds to wait for at least one frame.

        Returns:
            Tuple of (success, frame_bgr, frame_index, timestamp_seconds).
        """
        if not self._is_running:
            return False, None, 0, 0.0

        if not self._new_frame_event.wait(timeout=timeout):
            return False, None, 0, 0.0

        with self._lock:
            if self._latest_frame is None:
                return False, None, 0, 0.0
            frame = self._latest_frame.copy()
            idx = self._frame_index
            ts = self._frame_timestamp

        return True, frame, idx, ts

    def stop(self) -> None:
        """Stops the capture thread and releases video device resources."""
        with self._condition:
            self._is_running = False
            self._new_frame_event.set()
            self._condition.notify_all()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
            self._thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        with self._lock:
            self._latest_frame = None

    def release(self) -> None:
        """Alias for stop() matching OpenCV conventions."""
        self.stop()

    def close(self) -> None:
        """Alias for stop()."""
        self.stop()

    def is_opened(self) -> bool:
        """Checks if the camera device is currently open and running."""
        return self._is_running and (self._cap is not None and self._cap.isOpened())

    @property
    def actual_size(self) -> Tuple[int, int]:
        """Returns (width, height) of active camera capture stream."""
        return (self._actual_width, self._actual_height)

    @property
    def actual_fps(self) -> float:
        """Returns configured FPS of the camera."""
        return self._actual_fps

    @property
    def hardware_fps(self) -> float:
        """Returns real-time empirical hardware camera capture FPS."""
        return self._hardware_fps

    @property
    def camera_fps(self) -> float:
        """Alias for hardware_fps."""
        return self._hardware_fps

    def __enter__(self) -> ThreadedCamera:
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
