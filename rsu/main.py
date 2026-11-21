"""Run local videos through the real VLM and measure prediction throughput.

JSON Lines on stdout: one prediction per sampled frame, then a summary.
Startup messages go to stderr. Use --preview for local OpenCV playback alongside
inference. Network senders are not started. Sampling is fixed by default.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Callable, Protocol

import cv2
import numpy as np

from .decision_engine import CLASS_ORDER, DecisionConfig, DecisionEngine
from .frame_automation import CAMERA_IDS, DEFAULT_INTERVAL_SEC, FrameSampler
from .vision_review import DEFAULT_MODEL, Evidence, OpenAIFrameReviewer, ROI, TemporalReview, Verdict


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


def review_roi(value: str) -> ROI:
    try:
        x, y, width, height = (float(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use x,y,width,height as fractions of the frame") from exc
    if not (0 <= x < 1 and 0 <= y < 1 and 0 < width <= 1 - x + 1e-9
            and 0 < height <= 1 - y + 1e-9):
        raise argparse.ArgumentTypeError("the crop must fit inside the frame and have positive size")
    return x, y, width, height


class VideoPathAction(argparse.Action):
    """Collect repeated paths while keeping the original single-video attribute."""

    def __call__(self, parser, namespace, value, option_string=None):
        paths = namespace.videos + (value,)
        if len(paths) > 6:
            parser.error("at most six --video paths are supported")
        namespace.videos = paths
        namespace.video = paths[0]


def source_paths(args: argparse.Namespace) -> tuple[Path, ...]:
    paths = getattr(args, "videos", ()) or (args.video,)
    if len(paths) > 1:
        if args.streams not in (1, len(paths)):
            raise ValueError("with multiple --video paths, omit --streams or match the number of paths")
        return paths
    return paths * args.streams


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(videos=())
    parser.add_argument("--video", type=Path, required=True, action=VideoPathAction,
                        help="Path to a local video; repeat for up to six different clips")
    parser.add_argument("--streams", type=int, choices=range(1, 7), default=1,
                        help="Run 1–6 local copies of this video with one shared VLM (default: 1)")
    parser.add_argument("--camera-id", choices=CAMERA_IDS, default="CAM_01",
                        help="Camera identifier for single-stream runs")
    parser.add_argument("--interval-sec", type=positive_seconds,
                        help="Sampling interval (default: 1s fixed, 2s normal with vision review)")
    parser.add_argument("--openai-review", action="store_true",
                        help="Send ambiguous cases to OpenAI using up to three timestamped frames")
    parser.add_argument("--review-roi", type=review_roi, metavar="X,Y,W,H",
                        help="Normalized review crop; omit to send the full frame")
    parser.add_argument("--review-model", default=DEFAULT_MODEL,
                        help="OpenAI model used for vision review")
    parser.add_argument("--review-margin", type=float, default=DecisionConfig().vision_review_margin,
                        help="Maximum normalized top-two CLIP score gap for vision review")
    parser.add_argument("--max-frames", type=positive_count, help="Stop after this many predictions per stream")
    parser.add_argument("--no-realtime", action="store_true",
                        help="Process sampled frames without playback waits; measures processing capacity")
    parser.add_argument("--output-dir", type=Path, help="Save matching sampled frames as JPEGs")
    parser.add_argument("--only-class", choices=CLASS_ORDER,
                        help="Print/save only this class; all sampled frames still count in the summary")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"),
                        help="Override automatic model device selection")
    parser.add_argument("--preview", action="store_true",
                        help="Play the video in an OpenCV window while predicting in a worker")
    parser.add_argument("--loop", action="store_true",
                        help="Replay every clip continuously until quit/Ctrl-C or --max-frames")
    return parser


def sampling_interval(args: argparse.Namespace, reviewing: bool) -> float:
    if args.interval_sec is not None:
        return args.interval_sec
    return DecisionConfig().normal_interval_sec if reviewing else DEFAULT_INTERVAL_SEC


def emit_reviews(results, emit: Callable[[dict], None]) -> None:
    for evidence, verdict in results:
        emit({
            **evidence.target.metadata(),
            "type": "verification",
            "status": "resolved" if verdict.label is not None else "needs_human_review",
            "reviewed_class": verdict.label,
            "review_completed_at": datetime.now(timezone.utc).isoformat(),
            "reason": verdict.reason,
            "evidence_frames": [
                {"role": role, "frame_id": sample.frame_id,
                 "video_timestamp_sec": sample.video_timestamp_sec}
                for role, sample in evidence.ordered()
            ],
        })


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

    def record(self, prediction_ms: float, processing_ms: float, output_sec: float,
               interval_sec: float | None = None) -> float | None:
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
        self.processing_overruns += processing_ms > (interval_sec or self.interval_sec) * 1000
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
    reviewer: Callable[[Evidence], Verdict] | None = None,
) -> dict:
    """Run one camera. Only the CLI constructs the real, cached model instance."""
    interval = sampling_interval(args, reviewer is not None)
    stats = Throughput(interval)
    engine = reviews = None
    if reviewer is not None:
        engine = DecisionEngine(DecisionConfig(
            normal_interval_sec=interval,
            verify_interval_sec=min(DecisionConfig().verify_interval_sec, interval / 2),
            vision_review_margin=args.review_margin,
        ))
        reviews = TemporalReview(reviewer, clock=clock)
    status = "completed"
    printed = 0
    saved = 0
    first_video_sec = last_video_sec = None
    try:
        with FrameSampler(args.video, args.camera_id, interval, clock=clock) as camera:
            if args.output_dir:
                args.output_dir.mkdir(parents=True, exist_ok=True)
            if on_ready:
                on_ready()
            while args.max_frames is None or stats.count < args.max_frames:
                if reviews is not None:
                    emit_reviews(reviews.completed(), emit)
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
                decision_fields = {}
                if engine is not None and reviews is not None:
                    scores = prediction["scores"]
                    decision = engine.update(sample.camera_id, scores, defer_close_scores=True)
                    camera.set_interval(decision.sampling_interval_sec)
                    decision_fields = {
                        "verification_state": decision.verification_state,
                        "next_sampling_interval_sec": decision.sampling_interval_sec,
                        "priority": decision.priority,
                        "alert": decision.alert,
                        "event_id": decision.event_id,
                        "review_status": reviews.observe(sample, decision.review_required),
                    }
                matches = args.only_class is None or prediction["prediction"] == args.only_class
                image_path = None
                if matches and args.output_dir:
                    image_path = args.output_dir / f"{sample.frame_id}.jpg"
                    if not cv2.imwrite(str(image_path), sample.frame_bgr):
                        raise OSError(f"Could not save JPEG: {image_path}")
                    saved += 1
                output_sec = clock()
                processing_ms = (output_sec - frame_started) * 1000
                gap = stats.record(prediction_ms, processing_ms, output_sec, sample.sampling_interval_sec)
                if first_video_sec is None:
                    first_video_sec = sample.video_timestamp_sec
                last_video_sec = sample.video_timestamp_sec
                record = {
                    **sample.metadata(),
                    **prediction,
                    **decision_fields,
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
        if reviews is not None:
            emit_reviews(reviews.finish(wait=status == "completed"), emit)
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
        if reviews is not None:
            summary["adaptive_sampling"] = True
        emit(summary)
    return summary


def letterbox_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Fit the complete frame inside width × height; never crop or stretch it."""
    box_width, box_height = size
    height, width = frame.shape[:2]
    scale = min(box_width / width, box_height / height)
    scaled = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
    image = np.zeros((box_height, box_width, 3), dtype=np.uint8)
    left = (box_width - scaled.shape[1]) // 2
    top = (box_height - scaled.shape[0]) // 2
    image[top:top + scaled.shape[0], left:left + scaled.shape[1]] = scaled
    return image


def preview_image(
    frame: np.ndarray, video_sec: float, camera_id: str, latest: dict | None,
    *, size: tuple[int, int] = (960, 540), source_name: str | None = None,
    loop_index: int | None = None,
) -> np.ndarray:
    """Place status above the full, aspect-preserving image, never over footage."""
    width, _ = size
    compact = width < 640
    header = np.zeros((72 if compact else 94, width, 3), dtype=np.uint8)
    lines = [f"{camera_id} | video: {video_sec:.2f}s | Q / Esc: quit"]
    if compact:
        lines[0] = f"{camera_id} | video: {video_sec:.2f}s"
    if source_name:
        name = source_name if len(source_name) <= 20 else source_name[:17] + "..."
        lines[0] = f"{camera_id} | {name} | {video_sec:.1f}s"
    if loop_index is not None:
        lines[0] += f" L{loop_index}"
    if latest is None:
        lines += ["Waiting for first prediction..."]
    else:
        score = latest.get("confidence")
        score_text = f" | score: {score:.3f}" if score is not None else ""
        lines += [f"Latest result: {latest['prediction']}{score_text}",
                  f"AI frame: {latest['video_timestamp_sec']:.2f}s | "
                  f"inference: {latest['prediction_wall_ms']:.1f}ms"]
        if loop_index is not None:
            lines[-1] = (f"AI: {latest['video_timestamp_sec']:.2f}s L{latest['loop_index']} | "
                         f"{latest['prediction_wall_ms']:.1f}ms")
    for index, line in enumerate(lines):
        cv2.putText(header, line, (6 if compact else 12, (18 + index * 21) if compact else (24 + index * 27)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.39 if compact else 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack((header, letterbox_frame(frame, size)))


def run_preview(args: argparse.Namespace, predictor: Predictor, *, emit=emit_json,
                reviewer: Callable[[Evidence], Verdict] | None = None) -> dict:
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
                                          on_ready=source_ready, on_prediction=remember, reviewer=reviewer))
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
            cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        except cv2.error as exc:
            raise RuntimeError("Preview needs GUI-enabled opencv-python; replace "
                               "opencv-python-headless before using --preview") from exc
        window_created = True
        cv2.resizeWindow(window, 960, 634)
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
                cv2.imshow(window, preview_image(frame, video_sec, args.camera_id, prediction,
                                                source_name=args.video.name))
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


@dataclass
class StreamJob:
    frame: np.ndarray
    frame_id: str
    video_sec: float
    captured_at: str
    captured_clock: float
    queued_at: float
    decode_ms: float
    loop_index: int
    playback_elapsed_sec: float


@dataclass
class LocalStream:
    stream_id: str
    capture: cv2.VideoCapture
    fps: float
    stats: Throughput
    source: Path
    frame_index: int = -1
    clip_frame_index: int = -1
    loop_index: int = 0
    frame: np.ndarray | None = None
    video_sec: float = 0.0
    captured_at: str = ""
    captured_clock: float = 0.0
    decode_ms: float = 0.0
    last_sample_index: int = -1
    next_due_sec: float = 0.0
    sequence: int = 0
    pending: StreamJob | None = None
    active: bool = False
    eof: bool = False
    latest: dict | None = None
    dropped: int = 0
    cancelled: int = 0
    printed: int = 0
    saved: int = 0


def grab_stream_frame(stream: LocalStream, looping: bool) -> bool:
    """Advance one real frame, rewinding at EOF without resetting the run clock."""
    if stream.capture.grab():
        stream.frame_index += 1
        stream.clip_frame_index += 1
        return True
    if not looping:
        stream.eof = True
        return False
    if stream.clip_frame_index < 0:
        raise RuntimeError(f"No decodable frames to replay: {stream.source}")
    rewound = stream.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not rewound or not stream.capture.grab():
        # Some backends cannot seek. Reopen once, and fail explicitly if unreadable.
        stream.capture.release()
        stream.capture = cv2.VideoCapture(str(stream.source))
        if not stream.capture.isOpened() or not stream.capture.grab():
            raise RuntimeError(f"Could not restart video: {stream.source}")
    stream.loop_index += 1
    stream.frame_index += 1
    stream.clip_frame_index = 0
    return True


def run_streams(args: argparse.Namespace, predictor: Predictor, *, emit=emit_json) -> dict:
    """Local parallel-playback test with one model and one pending frame per stream.

    Local STREAM identifiers are not camera IDs and do not extend the network
    contract. Every completed prediction is emitted immediately, unless filtered.
    """
    interval_sec = sampling_interval(args, False)
    if args.no_realtime:
        raise ValueError("multiple streams require real-time mode")
    if args.preview and threading.current_thread() is not threading.main_thread():
        raise RuntimeError("OpenCV preview must run on the main thread")
    sources = source_paths(args)
    condition = threading.Condition()
    streams: list[LocalStream] = []
    producing = True
    errors: list[BaseException] = []
    status = "completed"
    worker = None
    window = f"RoadSense | {len(sources)} streams | Q / Esc: quit"
    window_created = False
    started = None

    def predict():
        cursor = 0
        try:
            while True:
                with condition:
                    condition.wait_for(lambda: any(s.pending is not None for s in streams) or not producing)
                    order = [(cursor + offset) % len(streams) for offset in range(len(streams))]
                    index = next((i for i in order if streams[i].pending is not None), None)
                    if index is None:
                        return
                    stream = streams[index]
                    job = stream.pending
                    stream.pending = None
                    stream.active = True
                    cursor = (index + 1) % len(streams)
                prediction_started = time.monotonic()
                prediction = predictor.predict(job.frame)
                prediction_ms = (time.monotonic() - prediction_started) * 1000
                matches = args.only_class is None or prediction["prediction"] == args.only_class
                image_path = None
                if matches and args.output_dir:
                    image_path = args.output_dir / f"{job.frame_id}.jpg"
                    if not cv2.imwrite(str(image_path), job.frame):
                        raise OSError(f"Could not save JPEG: {image_path}")
                finished = time.monotonic()
                processing_ms = job.decode_ms + (finished - prediction_started) * 1000
                with condition:
                    gap = stream.stats.record(prediction_ms, processing_ms, finished)
                    record = {
                        **prediction,
                        "type": "prediction", "stream_id": stream.stream_id,
                        "video": str(stream.source),
                        "frame_id": job.frame_id, "video_timestamp_sec": job.video_sec,
                        "loop_index": job.loop_index, "playback_elapsed_sec": job.playback_elapsed_sec,
                        "captured_at": job.captured_at, "sampling_interval_sec": interval_sec,
                        "output_at": datetime.now(timezone.utc).isoformat(),
                        "prediction_wall_ms": prediction_ms, "frame_processing_ms": processing_ms,
                        "queue_wait_ms": (prediction_started - job.queued_at) * 1000,
                        "capture_to_result_ms": (finished - job.captured_clock) * 1000,
                        "output_gap_sec": gap,
                    }
                    if image_path is not None:
                        record["image_path"] = str(image_path)
                        stream.saved += 1
                    stream.latest = record
                    stream.active = False
                if matches:
                    emit(record)
                    stream.printed += 1
        except BaseException as exc:
            with condition:
                errors.append(exc)
                condition.notify_all()

    try:
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(sources):
            capture = cv2.VideoCapture(str(source))
            fps = capture.get(cv2.CAP_PROP_FPS)
            streams.append(LocalStream(f"STREAM_{index + 1:02d}", capture, fps,
                                       Throughput(interval_sec), source))
            if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"Could not open stream with valid FPS: {source}")
        if args.preview:
            try:
                cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            except cv2.error as exc:
                raise RuntimeError("Preview needs GUI-enabled opencv-python") from exc
            window_created = True
            columns = min(3, len(sources))
            cv2.resizeWindow(window, columns * 320, math.ceil(len(sources) / columns) * 252)
        started = time.monotonic()
        worker = threading.Thread(target=predict, name="roadsense-shared-vlm")
        worker.start()
        last_displayed = None
        while True:
            with condition:
                if errors:
                    raise errors[0]
                if args.max_frames and all(s.eof or s.stats.count >= args.max_frames for s in streams):
                    break
            elapsed = time.monotonic() - started
            advanced = False
            for stream in streams:
                if stream.eof:
                    continue
                decoded_at = time.monotonic()
                target = max(0, int(elapsed * stream.fps))
                changed = False
                while stream.frame_index < target:
                    if not grab_stream_frame(stream, args.loop):
                        break
                    changed = True
                if changed and not stream.eof:
                    ok, frame = stream.capture.retrieve()
                    if not ok or frame is None:
                        raise RuntimeError(f"Could not decode {stream.stream_id}")
                    stream.frame = frame
                    timestamp = stream.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    stream.video_sec = timestamp if math.isfinite(timestamp) and timestamp > 0 \
                        else stream.clip_frame_index / stream.fps
                    stream.captured_at = datetime.now(timezone.utc).isoformat()
                    stream.captured_clock = time.monotonic()
                    stream.decode_ms = (stream.captured_clock - decoded_at) * 1000
                    advanced = True
                if stream.frame is not None and not stream.eof and elapsed >= stream.next_due_sec \
                        and stream.frame_index != stream.last_sample_index:
                    with condition:
                        reserved = stream.stats.count + int(stream.active) + int(stream.pending is not None)
                        if args.max_frames is None or reserved < args.max_frames or stream.pending is not None:
                            if stream.pending is not None:
                                stream.dropped += 1
                            stream.sequence += 1
                            stream.pending = StreamJob(
                                stream.frame, f"{stream.stream_id}-{stream.sequence:06d}", stream.video_sec,
                                stream.captured_at, stream.captured_clock, time.monotonic(), stream.decode_ms,
                                stream.loop_index, stream.frame_index / stream.fps,
                            )
                            stream.last_sample_index = stream.frame_index
                            condition.notify()
                    stream.next_due_sec = (math.floor(elapsed / interval_sec) + 1) * interval_sec
            with condition:
                latest = [s.latest for s in streams]
            if args.preview:
                signature = tuple(item["frame_id"] if item else None for item in latest)
                if all(s.frame is not None for s in streams) and (advanced or signature != last_displayed):
                    panels = [preview_image(s.frame, s.video_sec, s.stream_id, item, size=(320, 180),
                                            source_name=s.source.name,
                                            loop_index=s.loop_index if args.loop else None)
                              for s, item in zip(streams, latest)]
                    columns = min(3, len(panels))
                    while len(panels) % columns:
                        panels.append(np.zeros_like(panels[0]))
                    grid = np.vstack([np.hstack(panels[i:i + columns])
                                      for i in range(0, len(panels), columns)])
                    cv2.imshow(window, grid)
                    last_displayed = signature
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    status = "stopped"
                    break
                try:
                    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        status = "stopped"
                        break
                except cv2.error:
                    status = "stopped"
                    break
            if all(s.eof for s in streams):
                break
            time.sleep(0.001)
    except KeyboardInterrupt:
        status = "interrupted"
    except BaseException:
        status = "failed"
        raise
    finally:
        with condition:
            producing = False
            if status != "completed":
                for stream in streams:
                    if stream.pending is not None:
                        stream.pending = None
                        stream.cancelled += 1
            condition.notify_all()
        for stream in streams:
            stream.capture.release()
        if window_created:
            try:
                cv2.destroyWindow(window)
            except cv2.error:
                pass
        if worker is not None:
            worker.join()  # Drain bounded pending work at EOF; finish in-flight work on quit.
        summary = {
            "type": "summary", "status": "failed" if errors else status,
            "mode": "local_multistream_realtime", "video": str(args.video),
            "videos": [str(source) for source in sources],
            "streams": len(sources), "model_instances": 1, "preview": args.preview,
            "loop": args.loop,
            "predictions": sum(s.stats.count for s in streams),
            "per_stream": [{
                "stream_id": s.stream_id, "video": str(s.source), **s.stats.summary(),
                "source_exhausted": s.eof,
                "replays": s.loop_index,
                "printed_predictions": s.printed, "saved_images": s.saved,
                "replaced_pending_samples": s.dropped, "cancelled_pending_samples": s.cancelled,
            } for s in streams],
        }
        emit(summary)
    if errors:
        raise errors[0]
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        sources = source_paths(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.preview and args.no_realtime:
        parser.error("--preview cannot be combined with --no-realtime")
    if (len(sources) > 1 or args.loop) and args.no_realtime:
        parser.error("multiple streams and continuous replay require real-time mode")
    reviewer = None
    if args.openai_review:
        if args.loop:
            parser.error("--openai-review is not supported with continuous replay")
        if len(sources) > 1:
            parser.error("--openai-review requires a single stream")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            parser.error("--openai-review requires OPENAI_API_KEY on the RSU")
        if not 0 <= args.review_margin <= 1:
            parser.error("--review-margin must be between 0 and 1")
        reviewer = OpenAIFrameReviewer(api_key, model=args.review_model, roi=args.review_roi)
    elif args.review_roi is not None:
        parser.error("--review-roi requires --openai-review")
    try:
        # Validate every source before loading a large model or downloading weights.
        for source in dict.fromkeys(sources):
            with FrameSampler(source, args.camera_id, sampling_interval(args, reviewer is not None)):
                pass
        from .vlm import ClipPredictor

        print("Loading VLM and K30 prompts; first use may download model weights...", file=sys.stderr,
              flush=True)
        started = time.monotonic()
        predictor = ClipPredictor(device=args.device)
        load_ms = (time.monotonic() - started) * 1000
        print(f"VLM ready: device={predictor.device}, load_ms={load_ms:.1f}. Starting predictions.",
              file=sys.stderr, flush=True)
        if len(sources) > 1 or args.loop:
            run_streams(args, predictor)
        elif args.preview:
            run_preview(args, predictor, reviewer=reviewer)
        else:
            run_predictions(args, predictor, reviewer=reviewer)
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted.\n")
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        parser.exit(1, f"RSU prediction runner: {exc}\n")


if __name__ == "__main__":
    main()
