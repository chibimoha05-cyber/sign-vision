#!/usr/bin/env python3
"""Dataset Recorder for Sign Language Training (Team Members 2 & 3).

Interactive CLI/GUI utility that:
- Captures real-time hand movements for a designated sign label.
- Only commits samples to CSV when at least one hand is actively detected (presence_mask != (0, 0)).
- Exports deterministic CSV format:
    timestamp_ms, label, has_left, has_right, f_0, f_1, ..., f_125
- Includes pre-record countdown, live sample counter, and progress telemetry.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import time

# Auto-trampoline to project .venv if running with unconfigured system Python
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_venv_python = (
    os.path.join(_project_root, ".venv", "Scripts", "python.exe")
    if sys.platform == "win32"
    else os.path.join(_project_root, ".venv", "bin", "python")
)
if os.path.exists(_venv_python) and os.environ.get("_SIGN_VISION_TRAMPOLINE") != "1":
    try:
        import cv2  # noqa: F401
        import mediapipe  # noqa: F401
    except ImportError:
        os.environ["_SIGN_VISION_TRAMPOLINE"] = "1"
        os.execv(_venv_python, [_venv_python] + sys.argv)

from typing import Optional
import cv2
import numpy as np

# Ensure src/ is on sys.path
sys.path.insert(0, os.path.join(_project_root, "src"))

from sign_vision.capture import ThreadedCamera
from sign_vision.schema import FrameTrackingResult
from sign_vision.engine import SignHandTracker


def build_csv_header() -> list[str]:
    """Generates canonical CSV header matching the Phase 2.0 ML training specification."""
    header = ["timestamp_ms", "label", "has_left", "has_right"]
    # 142 features (68 left + 68 right + 6 global kinematics)
    for i in range(142):
        header.append(f"f_{i}")
    return header


def record_dataset(
    label: str,
    samples_needed: int = 100,
    output_path: str = "data/signs_dataset.csv",
    camera_id: int = 0,
    width: int = 640,
    height: int = 480,
    countdown_sec: float = 3.0,
    synthetic: bool = False,
    headless: bool = False,
) -> None:
    """Records real-time sign language landmark features into a standardized CSV."""
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    file_exists = out_file.exists() and out_file.stat().st_size > 0

    print("=" * 70)
    print("  SIGN LANGUAGE DATASET RECORDER (TEAM MEMBERS 2 & 3)")
    print("=" * 70)
    print(f"Target Sign Label:  '{label}'")
    print(f"Target Sample Count: {samples_needed} frames")
    print(f"Destination CSV:     {out_file}")
    print(f"Mode:                {'SYNTHETIC TEST' if synthetic else f'CAMERA {camera_id}'}")
    print("-" * 70)

    tracker = SignHandTracker(max_num_hands=2, enable_smoothing=True)

    cam: Optional[ThreadedCamera] = None
    if not synthetic:
        try:
            cam = ThreadedCamera(device_index=camera_id, width=width, height=height, mirror=True)
            cam.start()
        except Exception as e:
            print(f"Warning: Could not open camera {camera_id} ({e}). Switching to synthetic mode.")
            synthetic = True

    can_display = not headless and (os.environ.get("DISPLAY") is not None or sys.platform == "win32")
    window_name = f"Dataset Recorder - Recording '{label}'"

    if can_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1280, width), min(720, height))

    collected_samples = 0
    state = "COUNTDOWN" if countdown_sec > 0 else "RECORDING"
    start_time = time.time()
    countdown_end = start_time + countdown_sec

    # Open CSV file in append mode
    with open(out_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(build_csv_header())
            f.flush()

        frame_idx = 0
        try:
            while collected_samples < samples_needed:
                now = time.time()

                if synthetic:
                    # Synthetic test frame: create a black canvas and inject mock detection
                    frame_bgr = np.zeros((height, width, 3), dtype=np.uint8)
                    # Simulate synthetic hand
                    t = frame_idx * 0.1
                    cx = int(width * 0.5 + np.sin(t) * 30)
                    cy = int(height * 0.5)
                    cv2.circle(frame_bgr, (cx, cy), 60, (200, 200, 200), -1)
                else:
                    assert cam is not None
                    ok, frame_bgr = cam.read(timeout=1.0)
                    if not ok or frame_bgr is None:
                        continue

                result: FrameTrackingResult = tracker.process_frame(frame_bgr)
                annotated = tracker.draw_overlays(frame_bgr, result, draw_hud=False)

                # State Machine: COUNTDOWN vs RECORDING
                if state == "COUNTDOWN":
                    remaining = max(0.0, countdown_end - now)
                    if remaining <= 0:
                        state = "RECORDING"
                    else:
                        # Draw countdown overlay
                        cx, cy = width // 2, height // 2
                        overlay = annotated.copy()
                        cv2.rectangle(overlay, (cx - 250, cy - 80), (cx + 250, cy + 80), (10, 10, 15), -1)
                        cv2.addWeighted(overlay, 0.8, annotated, 0.2, 0, annotated)
                        cv2.putText(
                            annotated,
                            f"GET READY: '{label}'",
                            (cx - 190, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 230, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            annotated,
                            f"Starting in {remaining:3.1f}s...",
                            (cx - 150, cy + 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                elif state == "RECORDING":
                    # Record row ONLY when at least one hand is detected
                    if result.has_any_hand or synthetic:
                        # Vector indices: 0 is has_left, 1 is has_right, 2..143 are the 142 features
                        vec = result.feature_vector
                        has_l = int(vec[0])
                        has_r = int(vec[1])
                        # If synthetic mode in test has no palm, synthesize a dummy valid coordinate
                        if synthetic and not result.has_any_hand:
                            has_r = 1
                            coords = [float(np.sin(frame_idx * 0.05))] * 142
                        else:
                            coords = [float(f) for f in vec[2:144]]

                        row = [int(result.timestamp_ms), label, has_l, has_r] + coords
                        writer.writerow(row)
                        f.flush()
                        collected_samples += 1

                    # Recording HUD Banner
                    rx, ry, rw, rh = 16, 16, 420, 95
                    overlay = annotated.copy()
                    cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), (15, 18, 22), -1)
                    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)
                    cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (40, 60, 220), 2, cv2.LINE_AA)

                    # Pulsing red recording dot
                    pulse = int(255 * (0.6 + 0.4 * np.sin(time.time() * 6.0)))
                    cv2.circle(annotated, (rx + 24, ry + 28), 8, (30, 30, pulse), -1, cv2.LINE_AA)

                    cv2.putText(
                        annotated,
                        f"REC: '{label}' ({collected_samples}/{samples_needed})",
                        (rx + 42, ry + 34),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    # Progress Bar
                    bar_w = rw - 40
                    bar_filled = int(bar_w * (collected_samples / max(1, samples_needed)))
                    cv2.rectangle(annotated, (rx + 20, ry + 52), (rx + 20 + bar_w, ry + 66), (40, 45, 55), -1)
                    cv2.rectangle(annotated, (rx + 20, ry + 52), (rx + 20 + bar_filled, ry + 66), (0, 220, 120), -1)

                    # Status Text
                    status_txt = "Capturing hand pose..." if result.has_any_hand else "WAITING: Place hand in frame"
                    status_col = (0, 255, 150) if result.has_any_hand else (60, 120, 255)
                    cv2.putText(
                        annotated,
                        status_txt,
                        (rx + 20, ry + 84),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        status_col,
                        1,
                        cv2.LINE_AA,
                    )

                frame_idx += 1

                if can_display:
                    cv2.imshow(window_name, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q"), ord("Q")):
                        print("\nRecording cancelled by user.")
                        break
                    elif key == 32:  # SPACE bar
                        state = "PAUSED" if state == "RECORDING" else "RECORDING"
                        print(f"Recording state: {state}")

        finally:
            if cam is not None:
                cam.stop()
            tracker.close()
            if can_display:
                cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print(f"  RECORDING COMPLETE: {collected_samples} samples collected for '{label}'")
    print(f"  Output saved to: {out_file}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="SignVision Dataset Recorder for ML Training")
    parser.add_argument("--label", type=str, required=True, help="Sign name / label (e.g., 'HELLO', 'YES', 'A')")
    parser.add_argument("--samples", type=int, default=100, help="Number of active hand samples to record (default: 100)")
    parser.add_argument("--output", type=str, default="data/signs_dataset.csv", help="Destination CSV filepath")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device ID (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    parser.add_argument("--delay", type=float, default=3.0, help="Countdown delay in seconds before recording (default: 3.0)")
    parser.add_argument("--synthetic", action="store_true", help="Record with synthetic generated samples for testing")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI window")

    args = parser.parse_args()

    record_dataset(
        label=args.label,
        samples_needed=args.samples,
        output_path=args.output,
        camera_id=args.camera,
        width=args.width,
        height=args.height,
        countdown_sec=args.delay,
        synthetic=args.synthetic,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
