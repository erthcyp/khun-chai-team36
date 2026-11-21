"""Run a local video through the real VLM and measure prediction throughput.

JSON Lines on stdout: one prediction per sampled frame, then a summary.
Startup messages go to stderr. This runner does not start a preview or network
senders. Fixed-rate sampling is intentional for an initial throughput test.
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
            while args.max_frames is None or stats.count < args.max_frames:
                frame_started = clock()
                sample = camera.read_due(realtime=not args.no_realtime)
                if sample is None:
                    if camera.exhausted:
                        break
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
                if matches:
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


def main(argv: list[str] | None = None) -> None:
    parser = make_parser()
    args = parser.parse_args(argv)
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
        run_predictions(args, predictor)
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted during startup.\n")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"RSU prediction runner: {exc}\n")


if __name__ == "__main__":
    main()
