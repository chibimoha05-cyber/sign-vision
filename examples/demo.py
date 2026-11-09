#!/usr/bin/env python3
"""Live Inspector and Real-Time HUD Diagnostic Viewer for hand tracking.

Captures hardware camera input or runs a synthetic benchmark, processes frames
through the SignHandTracker pipeline, and renders a Cyber HUD overlay with
feature vector telemetry, kinematics, and backend/exposure controls.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

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

sys.path.insert(0, os.path.join(_project_root, "src"))

from sign_vision.capture import ThreadedCamera
from sign_vision.transforms import HandStabilizer
from sign_vision.engine import SignHandTracker


def generate_synthetic_hand_frame(width: int, height: int, frame_idx: int) -> np.ndarray:
    """Generates an organic synthetic hand silhouette on a dark canvas for benchmarking."""
    frame = np.full((height, width, 3), (25, 28, 35), dtype=np.uint8)

    t = frame_idx * 0.05
    cx = int(width * 0.5 + np.sin(t) * 40)
    cy = int(height * 0.55 + np.cos(t) * 30)

    cv2.circle(frame, (cx, cy), 55, (190, 205, 235), -1, cv2.LINE_AA)

    finger_angles = [-0.6, -0.3, 0.0, 0.3, 0.6]
    finger_lens = [75, 95, 105, 90, 70]
    for angle, flen in zip(finger_angles, finger_lens):
        fx = int(cx + np.sin(angle) * flen)
        fy = int(cy - np.cos(angle) * flen)
        cv2.line(frame, (cx, cy - 15), (fx, fy), (180, 195, 225), 18, cv2.LINE_AA)
        cv2.circle(frame, (fx, fy), 10, (180, 195, 225), -1, cv2.LINE_AA)

    cv2.rectangle(frame, (cx - 25, cy + 25), (cx + 25, cy + 110), (170, 185, 215), -1)

    return frame


def render_feature_vector_hud(
    frame: np.ndarray,
    feature_vector: np.ndarray,
    x: int = 12,
    y: int = 106,
    width: int = 350,
    height: int = 110,
) -> None:
    """Renders a visual mini-bar telemetry display of the 144-float feature vector."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (15, 18, 22), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 70, 85), 1, cv2.LINE_AA)

    cv2.putText(
        frame,
        "FEATURE VECTOR PREVIEW (144-D)",
        (x + 10, y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )

    pres_str = f"Pres: L={int(feature_vector[0])} R={int(feature_vector[1])}"
    cv2.putText(frame, pres_str, (x + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 210, 220), 1, cv2.LINE_AA)

    bars_start_x = x + 10
    bars_y_center = y + 75
    sample_values = feature_vector[2:26]
    bar_w = max(2, (width - 25) // len(sample_values))

    for i, val in enumerate(sample_values):
        bx = bars_start_x + i * bar_w
        bar_h = int(np.clip(val, -1.0, 1.0) * 22.0)
        color = (255, 210, 0) if val >= 0 else (50, 120, 255)
        if bar_h >= 0:
            cv2.rectangle(frame, (bx, bars_y_center - bar_h), (bx + bar_w - 1, bars_y_center), color, -1)
        else:
            cv2.rectangle(frame, (bx, bars_y_center), (bx + bar_w - 1, bars_y_center - bar_h), color, -1)

    cv2.line(frame, (bars_start_x, bars_y_center), (bars_start_x + len(sample_values) * bar_w, bars_y_center), (100, 110, 120), 1)

    cv2.putText(
        frame,
        "[B]ackend | [E]anti-blur | [[/]]exp | [S]mooth | [K]inematics | [Q]uit",
        (x + 10, y + 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.31,
        (130, 140, 150),
        1,
        cv2.LINE_AA,
    )


def run_inspector(
    camera_id: int = 0,
    width: int = 640,
    height: int = 480,
    target_fps: int = 60,
    benchmark_synthetic: bool = False,
    max_frames: Optional[int] = None,
    headless: bool = False,
    enable_smoothing: bool = True,
    enhance_lighting: bool = False,
    sharpen_finger_edges: bool = False,
    anti_blur_mode: bool = True,
    preferred_backend: str = "auto",
) -> None:
    """Main execution loop for live camera inspection or synthetic benchmarking."""
    print("=" * 70)
    print("  SIGN LANGUAGE TRANSLATION - PHASE 3.0 HIGH-VELOCITY DISAMBIGUATION HUD")
    print("=" * 70)
    print(f"Mode: {'SYNTHETIC BENCHMARK' if benchmark_synthetic else f'CAMERA {camera_id}'}")
    print(f"Native Resolution: {width}x{height} @ {target_fps} FPS | Anti-Blur: {anti_blur_mode}")
    print(f"1-Euro Smoothing: {enable_smoothing} | Preferred Backend: {preferred_backend}")

    can_display = not headless and (os.environ.get("DISPLAY") is not None or sys.platform == "win32")

    print(f"Display Window: {'Enabled' if can_display else 'Headless / Benchmark-only'}")
    print("-" * 70)

    tracker = SignHandTracker(
        max_num_hands=2,
        min_detection_confidence=0.40,
        min_presence_confidence=0.35,
        min_tracking_confidence=0.35,
        enable_smoothing=enable_smoothing,
        smoothing_mode="one_euro",
        inference_width=width,
        enhance_lighting=enhance_lighting,
        sharpen_finger_edges=sharpen_finger_edges,
        static_image_mode_on_drift=True,
        enforce_bone_rigidity=False,
        max_coast_frames=1,
        preferred_backend=preferred_backend,
    )
    print(f"Active Vision Engine: {tracker.active_backend_name.upper()} (Inference width: {width}px)")

    cam: Optional[ThreadedCamera] = None
    if not benchmark_synthetic:
        try:
            cam = ThreadedCamera(
                device_index=camera_id,
                width=width,
                height=height,
                target_fps=target_fps,
                mirror=True,
                anti_blur_mode=anti_blur_mode,
            )
            cam.start()
            print("Zero-lag camera capture thread successfully spawned with Unique Frame Lock.")
        except Exception as e:
            print(f"Could not open hardware camera {camera_id} ({e}).")
            print("Falling back to synthetic benchmark mode automatically.")
            benchmark_synthetic = True

    frame_idx = 0
    last_consumed_idx = -1
    fps_history = []
    latency_history = []
    show_kinematics = True
    window_name = "SignVision 3.0 - High-Velocity Hand Disambiguation HUD"

    display_w = max(640, min(1280, width * 3 // 2 if width <= 640 else width))
    display_h = max(480, min(960, height * 3 // 2 if height <= 480 else height))

    if can_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_w, display_h)

    last_time = time.perf_counter()
    modes = ["one_euro", "fast", "unfiltered"]

    try:
        while True:
            if benchmark_synthetic:
                frame_bgr = generate_synthetic_hand_frame(width, height, frame_idx)
            else:
                assert cam is not None
                ok, frame_bgr, new_idx, _ = cam.read_new_frame(last_consumed_index=last_consumed_idx, timeout=0.1)
                if not ok or frame_bgr is None:
                    continue
                last_consumed_idx = new_idx

            result = tracker.process_frame(frame_bgr)
            latency_history.append(result.inference_latency_ms)

            cam_fps = cam.hardware_fps if cam is not None else 0.0
            annotated = tracker.draw_overlays(
                frame_bgr,
                result,
                draw_hud=True,
                camera_fps=cam_fps,
                draw_kinematics=show_kinematics,
            )
            render_feature_vector_hud(annotated, result.feature_vector)

            now = time.perf_counter()
            dt = now - last_time
            last_time = now
            cur_fps = 1.0 / dt if dt > 0 else 0.0
            fps_history.append(cur_fps)
            if len(fps_history) > 60:
                fps_history.pop(0)
                latency_history.pop(0)

            avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0.0
            avg_lat = sum(latency_history) / len(latency_history) if latency_history else 0.0
            exp_str = f" | Exp:{cam.exposure_val}" if (cam and cam.anti_blur_mode) else ""
            fps_str = f"Pipe:{avg_fps:4.1f} FPS | Cam:{cam_fps:4.1f} FPS | Lat:{avg_lat:4.1f}ms{exp_str}"
            cv2.putText(annotated, fps_str, (annotated.shape[1] - 380, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 180), 1, cv2.LINE_AA)

            frame_idx += 1

            if can_display:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    print("\nExiting per user request.")
                    break
                elif key in (ord("b"), ord("B")):
                    new_backend = tracker.switch_backend()
                    print(f"Switched Backend Engine: {new_backend.upper()}")
                elif key in (ord("e"), ord("E")):
                    if cam is not None:
                        new_state = cam.set_anti_blur(not cam.anti_blur_mode)
                        print(f"Toggled Anti-Blur Fast Shutter: {'ENABLED' if new_state else 'DISABLED'}")
                elif key == ord("["):
                    if cam is not None:
                        new_exp = cam.adjust_exposure(-20)
                        print(f"Exposure Time Decreased: {new_exp} (Crisper shutter)")
                elif key == ord("]"):
                    if cam is not None:
                        new_exp = cam.adjust_exposure(+20)
                        print(f"Exposure Time Increased: {new_exp} (Brighter)")
                elif key in (ord("s"), ord("S")):
                    if tracker.stabilizer is not None:
                        current_mode = tracker.stabilizer.mode
                        next_idx = (modes.index(current_mode) + 1) % len(modes)
                        new_mode = modes[next_idx]
                        tracker.stabilizer.set_mode(new_mode)
                        print(f"Switched Smoothing Mode: {new_mode.upper()}")
                elif key in (ord("l"), ord("L")):
                    tracker.enhance_lighting = not tracker.enhance_lighting
                    print(f"Toggled Fast Lighting Boost: {'ENABLED' if tracker.enhance_lighting else 'DISABLED'}")
                elif key in (ord("k"), ord("K")):
                    show_kinematics = not show_kinematics
                    print(f"Toggled Biomechanical Kinematics Dashboard: {'ON' if show_kinematics else 'OFF'}")
                elif key in (ord("n"), ord("N")):
                    print("\n" + "=" * 50)
                    print(f"FEATURE VECTOR SNAPSHOT (Frame #{result.frame_index}):")
                    print(f"Left Present: {result.presence_mask[0]} | Right Present: {result.presence_mask[1]}")
                    print(f"Vector Shape: {result.feature_vector.shape} (dtype: {result.feature_vector.dtype})")
                    print(f"Slice [0:10]: {np.round(result.feature_vector[:10], 4)}")
                    if result.has_any_hand:
                        active = result.right_hand or result.left_hand
                        if active and active.finger_curls is not None:
                            print(f"Finger Curls: {np.round(active.finger_curls, 3)} [Thumb, Idx, Mid, Ring, Pinky]")
                    print("=" * 50 + "\n")

            if max_frames is not None and frame_idx >= max_frames:
                print(f"\nReached target of {max_frames} frames.")
                break

    finally:
        if cam is not None:
            cam.stop()
        tracker.close()
        if can_display:
            cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print("  SESSION BENCHMARK REPORT (PHASE 3.0)")
    print("=" * 70)
    print(f"Total Frames Processed: {frame_idx}")
    print(f"Active Engine:          {tracker.active_backend_name.upper()}")
    print(f"Re-detect Triggers:     {tracker.redetect_trigger_count}")
    if fps_history:
        print(f"Average Pipeline FPS:    {sum(fps_history)/len(fps_history):.2f}")
    if latency_history:
        print(f"Average Inference Time:  {sum(latency_history)/len(latency_history):.2f} ms")
        print(f"Min / Max Latency:       {min(latency_history):.2f} ms / {max(latency_history):.2f} ms")
    if cam is not None:
        print(f"Hardware Camera FPS:     {cam.hardware_fps:.2f} FPS")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="SignVision Live Inspector & HUD Diagnostic Tool")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device ID (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    parser.add_argument("--fps", type=int, default=60, help="Target FPS (default: 60)")
    parser.add_argument("--benchmark-synthetic", action="store_true", help="Run with synthetic generated frames")
    parser.add_argument("--frames", type=int, default=None, help="Stop after N frames (useful for benchmarks)")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI window")
    parser.add_argument("--no-smoothing", action="store_true", help="Disable smoothing")
    parser.add_argument("--enhance-lighting", action="store_true", help="Enable fast lighting / contrast boost")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "tasks", "solutions", "mediapipe", "rtmpose"], help="Preferred backend engine")

    args = parser.parse_args()

    frames = args.frames
    if args.benchmark_synthetic and frames is None:
        frames = 100

    run_inspector(
        camera_id=args.camera,
        width=args.width,
        height=args.height,
        target_fps=args.fps,
        benchmark_synthetic=args.benchmark_synthetic,
        max_frames=frames,
        headless=args.headless,
        enable_smoothing=not args.no_smoothing,
        enhance_lighting=args.enhance_lighting,
        preferred_backend=args.backend,
    )


if __name__ == "__main__":
    main()
