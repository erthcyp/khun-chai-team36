"""Run a local video through the real VLM and measure prediction throughput.

JSON Lines on stdout: one prediction per sampled frame, then a summary.
Startup messages go to stderr. Use --preview for local OpenCV playback alongside
inference. Network senders are not started. Sampling uses a fixed interval.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import threading
import time
from typing import Callable, Protocol

import cv2
import numpy as np

from .decision_engine import CLASS_ORDER
from .frame_automation import CAMERA_IDS, DEFAULT_INTERVAL_SEC, FrameSampler


class Predictor(Protocol):
    def predict(self, frame: np.ndarray) -> dict: ...


def positive_seconds(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def positive_count(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Path to a local video file")
    parser.add_argument("--camera-id", choices=CAMERA_IDS, default="CAM_01")
    parser.add_argument("--interval-sec", type=positive_seconds, default=DEFAULT_INTERVAL_SEC)
    parser.add_argument("--max-frames", type=positive_count, help="Stop after this many predictions")
    parser.add_argument("--no-realtime", action="store_true",
                        help="Process sampled frames without playback waits; measures processing capacity")
    parser.add_argument("--output-dir", type=Path, help="Save matching sampled frames as JPEGs")
    parser.add_argument("--only-class", choices=CLASS_ORDER,
                        help="Print/save only this class; all sampled frames still count in the summary")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"),
                        help="Override automatic model device selection")
    parser.add_argument("--preview", action="store_true",
                        help="Play the video in an OpenCV window while predicting in a worker")
    return parser


def emit_json(record: dict) -> None:
    print(json.dumps(record, allow_nan=False), flush=True)


@dataclass
class Throughput:
    interval_sec: float
    count: int = 0
    total_prediction_ms: float = 0.0
    total_processing_ms: float = 0.0
    max_prediction_ms: float = 0.0
    processing_overruns: int = 0
    first_output_sec: float | None = None
    last_output_sec: float | None = None
    max_output_gap_sec: float | None = None
    # Bound metric history as well as image lifetime for long-running clips.
    prediction_window: deque = field(default_factory=lambda: deque(maxlen=1000))

    def record(self, prediction_ms: float, processing_ms: float, output_sec: float) -> float | None:
        gap = None if self.last_output_sec is None else output_sec - self.last_output_sec
        if self.first_output_sec is None:
            self.first_output_sec = output_sec
        self.last_output_sec = output_sec
        if gap is not None:
            self.max_output_gap_sec = max(self.max_output_gap_sec or 0.0, gap)
        self.count += 1
        self.total_prediction_ms += prediction_ms
        self.total_processing_ms += processing_ms
        self.max_prediction_ms = max(self.max_prediction_ms, prediction_ms)
        self.processing_overruns += processing_ms > self.interval_sec * 1000
        self.prediction_window.append(prediction_ms)
        return gap

    def summary(self) -> dict:
        span = None
        if self.count > 1:
            span = self.last_output_sec - self.first_output_sec
        return {
            "predictions": self.count,
            "requested_interval_sec": self.interval_sec,
            "target_output_fps": 1 / self.interval_sec,
            # N-1 output gaps avoids inflating FPS from the immediate first sample.
            "observed_output_fps": (self.count - 1) / span if span and span > 0 else None,
            "model_call_rate_fps": self.count * 1000 / self.total_prediction_ms
            if self.total_prediction_ms else None,
            "mean_prediction_ms": self.total_prediction_ms / self.count if self.count else None,
            "p95_prediction_ms": float(np.percentile(self.prediction_window, 95))
            if self.prediction_window else None,
            "p95_window_frames": len(self.prediction_window),
            "max_prediction_ms": self.max_prediction_ms if self.count else None,
            "mean_frame_processing_ms": self.total_processing_ms / self.count if self.count else None,
            "frame_processing_overruns": self.processing_overruns,
            "max_output_gap_sec": self.max_output_gap_sec,
        }


def run_predictions(
    args: argparse.Namespace,
    predictor: Predictor,
    *,
    emit: Callable[[dict], None] = emit_json,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    stop_event: threading.Event | None = None,
    on_ready: Callable[[], None] | None = None,
    on_prediction: Callable[[dict], None] | None = None,
) -> dict:
    """Run one camera. Only the CLI constructs the real, cached model instance."""
    stats = Throughput(args.interval_sec)
    status = "completed"
    printed = 0
    saved = 0
    first_video_sec = last_video_sec = None
    try:
        with FrameSampler(args.video, args.camera_id, args.interval_sec, clock=clock) as camera:
            if args.output_dir:
                args.output_dir.mkdir(parents=True, exist_ok=True)
            if on_ready:
                on_ready()
            while args.max_frames is None or stats.count < args.max_frames:
                if stop_event is not None and stop_event.is_set():
                    status = "stopped"
                    break
                frame_started = clock()
                sample = camera.read_due(realtime=not args.no_realtime)
                if sample is None:
                    if camera.exhausted:
                        break
                    if stop_event is not None:
                        stop_event.wait(camera.seconds_until_due())
                    else:
                        sleep(camera.seconds_until_due())
                    continue
                prediction_started = clock()
                prediction = predictor.predict(sample.frame_bgr)
                prediction_ms = (clock() - prediction_started) * 1000
                matches = args.only_class is None or prediction["prediction"] == args.only_class
                image_path = None
                if matches and args.output_dir:
                    image_path = args.output_dir / f"{sample.frame_id}.jpg"
                    if not cv2.imwrite(str(image_path), sample.frame_bgr):
                        raise OSError(f"Could not save JPEG: {image_path}")
                    saved += 1
                output_sec = clock()
                processing_ms = (output_sec - frame_started) * 1000
                gap = stats.record(prediction_ms, processing_ms, output_sec)
                if first_video_sec is None:
                    first_video_sec = sample.video_timestamp_sec
                last_video_sec = sample.video_timestamp_sec
                record = {
                    **sample.metadata(),
                    **prediction,
                    "type": "prediction",
                    "output_at": datetime.now(timezone.utc).isoformat(),
                    "prediction_wall_ms": prediction_ms,
                    "frame_processing_ms": processing_ms,
                    "output_gap_sec": gap,
                }
                if image_path is not None:
                    record["image_path"] = str(image_path)
                if on_prediction:
                    on_prediction(record)
                if matches:
                    emit(record)
                    printed += 1
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception:
        status = "failed"
        raise
    finally:
        summary = {
            "type": "summary",
            "status": status,
            "mode": "offline" if args.no_realtime else "realtime",
            "preview": args.preview,
            "video": str(args.video),
            "camera_id": args.camera_id,
            **stats.summary(),
            "printed_predictions": printed,
            "saved_images": saved,
            "first_video_timestamp_sec": first_video_sec,
            "last_video_timestamp_sec": last_video_sec,
        }
        emit(summary)
    return summary


def preview_image(frame: np.ndarray, video_sec: float, camera_id: str, latest: dict | None) -> np.ndarray:
    """Draw on a bounded-size display copy; prediction input stays untouched."""
    height, width = frame.shape[:2]
    scale = min(1.0, 960 / width, 640 / height)
    image = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
    cv2.rectangle(image, (0, 0), (image.shape[1], 94), (0, 0, 0), -1)
    lines = [f"{camera_id} | video: {video_sec:.2f}s | Q / Esc: quit"]
    if latest is None:
        lines += ["Waiting for first prediction..."]
    else:
        score = latest.get("confidence")
        score_text = f" | score: {score:.3f}" if score is not None else ""
        lines += [f"Latest result: {latest['prediction']}{score_text}",
                  f"AI frame: {latest['video_timestamp_sec']:.2f}s | "
                  f"inference: {latest['prediction_wall_ms']:.1f}ms"]
    for index, line in enumerate(lines):
        cv2.putText(image, line, (12, 24 + index * 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def run_preview(args: argparse.Namespace, predictor: Predictor, *, emit=emit_json) -> dict:
    """Keep macOS GUI events on the main thread, with serialized VLM work off-thread.

    Preview and inference use separate local file readers, synchronized at start.
    Only the latest prediction metadata is retained; no frames are queued.
    """
    if args.no_realtime:
        raise ValueError("--preview cannot be combined with --no-realtime")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("OpenCV preview must run on the main thread")
    capture = cv2.VideoCapture(str(args.video))
    window = f"RoadSense | {args.camera_id} | {args.video.name}"
    stop = threading.Event()
    ready = threading.Event()
    done = threading.Event()
    lock = threading.Lock()
    latest = None
    started_at = None
    result = []
    errors = []

    def source_ready():
        nonlocal started_at
        started_at = time.monotonic()
        ready.set()

    def remember(record):
        nonlocal latest
        with lock:
            latest = record

    def predict():
        try:
            result.append(run_predictions(args, predictor, emit=emit, stop_event=stop,
                                          on_ready=source_ready, on_prediction=remember))
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    worker = threading.Thread(target=predict, name="roadsense-inference")
    window_created = False
    worker_started = False
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Could not open preview with valid FPS: {args.video}")
        try:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            raise RuntimeError("Preview needs GUI-enabled opencv-python; replace "
                               "opencv-python-headless before using --preview") from exc
        window_created = True
        cv2.resizeWindow(window, 960, 540)
        worker.start()
        worker_started = True
        index = -1
        at_eof = False
        frame = None
        last_displayed_result = None
        video_sec = 0.0
        while not done.is_set():
            advanced = False
            if ready.is_set() and not at_eof:
                target = max(0, int((time.monotonic() - started_at) * fps))
                while index < target:
                    if not capture.grab():
                        at_eof = True
                        break
                    index += 1
                    advanced = True
                if advanced and not at_eof:
                    ok, decoded = capture.retrieve()
                    if not ok or decoded is None:
                        raise RuntimeError("Could not decode a preview frame")
                    frame = decoded
                    timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    video_sec = timestamp if math.isfinite(timestamp) and timestamp > 0 else index / fps
            with lock:
                prediction = latest
            if frame is not None and (advanced or prediction is not last_displayed_result):
                cv2.imshow(window, preview_image(frame, video_sec, args.camera_id, prediction))
                last_displayed_result = prediction
            key = cv2.waitKey(10) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break  # Closing the native window can remove it before this poll.
    finally:
        stop.set()
        capture.release()
        if window_created:
            try:
                cv2.destroyWindow(window)
            except cv2.error:
                pass  # The user may already have closed the native window.
        if worker_started:
            worker.join()  # Finish an in-flight prediction before exiting.
    if errors:
        raise errors[0]
    return result[0]


def main(argv: list[str] | None = None) -> None:
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.preview and args.no_realtime:
        parser.error("--preview cannot be combined with --no-realtime")
    try:
        # Validate the video before loading a large model or downloading weights.
        with FrameSampler(args.video, args.camera_id, args.interval_sec):
            pass
        from .vlm import ClipPredictor

        print("Loading VLM and K30 prompts; first use may download model weights...", file=sys.stderr,
              flush=True)
        started = time.monotonic()
        predictor = ClipPredictor(device=args.device)
        load_ms = (time.monotonic() - started) * 1000
        print(f"VLM ready: device={predictor.device}, load_ms={load_ms:.1f}. Starting predictions.",
              file=sys.stderr, flush=True)
        if args.preview:
            run_preview(args, predictor)
        else:
            run_predictions(args, predictor)
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted.\n")
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        parser.exit(1, f"RSU prediction runner: {exc}\n")


if __name__ == "__main__":
    main()
