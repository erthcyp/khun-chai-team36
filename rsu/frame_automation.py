"""OpenCV sampling for local simulated CCTV clips (no model or network calls).

One FrameSampler belongs to one camera. Call read_due() from the RSU loop,
pass sample.frame_bgr to the VLM, then call set_interval() with its decision.
Run `python -m rsu.frame_automation --help` for an extraction smoke test.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Callable, Iterator

import cv2
import numpy as np


DEFAULT_INTERVAL_SEC = 1.0
CAMERA_IDS = ("CAM_01", "CAM_02", "CAM_03")


def _validate_interval(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("sampling interval must be finite and greater than zero")
    return value


@dataclass(frozen=True)
class FrameSample:
    camera_id: str
    frame_id: str
    video_timestamp_sec: float
    captured_at: str
    sampling_interval_sec: float
    frame_bgr: np.ndarray

    def metadata(self) -> dict:
        """Metadata only; the BGR image stays in memory for inference/JPEG encoding."""
        return {
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "video_timestamp_sec": self.video_timestamp_sec,
            "captured_at": self.captured_at,
            "sampling_interval_sec": self.sampling_interval_sec,
        }


class FrameSampler:
    """Sample a local video on an independent, monotonic playback clock.

    The first frame is due immediately. Intervals measure video seconds. OpenCV
    presentation timestamps are used when available, with frame index / FPS as
    a fallback. Valid FPS metadata is required. Frames between samples are
    grabbed without converting them to BGR. Sampling is accurate to one source
    frame, and cannot exceed the source frame rate.

    read_due() never sleeps. If the caller falls behind by at least one sampling
    interval, it skips to current playback time instead of building a backlog.
    A slow model can still reduce throughput; this class does not run inference.
    Calls for a given instance must be serialized by the RSU scheduler.
    """

    def __init__(
        self,
        source: str | Path,
        camera_id: str = "CAM_01",
        interval_sec: float = DEFAULT_INTERVAL_SEC,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if camera_id not in CAMERA_IDS:
            raise ValueError(f"camera_id must be one of {CAMERA_IDS}")
        self._interval = _validate_interval(interval_sec)
        self.camera_id = camera_id
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        self._capture = cv2.VideoCapture(str(source))
        if not self._capture.isOpened():
            self._capture.release()
            raise ValueError(f"OpenCV could not open video: {source}")
        self.fps = self._capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(self.fps) or self.fps <= 0:
            self._capture.release()
            raise ValueError(f"Video needs valid FPS metadata: {source}")
        self._clock = clock
        self._started_at = clock()
        self._next_due_sec = 0.0
        self._last_sample_sec: float | None = None
        self._last_frame_sec = -1.0
        self._frame_index = -1
        self._sample_count = 0
        self._closed = False
        self.exhausted = False

    @property
    def interval_sec(self) -> float:
        return self._interval

    def set_interval(self, interval_sec: float) -> None:
        """Apply a local decision to this camera, relative to its last sample.

        Hysteresis/cooldown belongs in the decision engine. An unchanged interval
        does not postpone the next sample.
        """
        interval_sec = _validate_interval(interval_sec)
        if interval_sec == self._interval:
            return
        self._interval = interval_sec
        if self._last_sample_sec is not None:
            self._next_due_sec = self._last_sample_sec + self._interval

    def seconds_until_due(self) -> float:
        return max(0.0, self._next_due_sec - (self._clock() - self._started_at))

    def read_due(self, *, realtime: bool = True) -> FrameSample | None:
        """Return a due sample, or None when not due/finished (see exhausted).

        Set realtime=False for offline extraction/benchmarking without sleeps.
        Decode errors and EOF both end the source; no predictions are fabricated.
        """
        if self._closed:
            return None
        elapsed = max(0.0, self._clock() - self._started_at)
        if realtime and elapsed + 1e-9 < self._next_due_sec:
            return None
        target = self._next_due_sec
        if realtime and elapsed - target >= self._interval:
            target = elapsed

        while self._capture.grab():
            self._frame_index += 1
            timestamp = self._capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not math.isfinite(timestamp) or timestamp < 0 or timestamp <= self._last_frame_sec:
                timestamp = max(self._frame_index / self.fps, self._last_frame_sec + 1 / self.fps)
            self._last_frame_sec = timestamp
            if timestamp + 1e-9 < target:
                continue
            ok, frame = self._capture.retrieve()
            if not ok or frame is None or frame.size == 0:
                self._finish()
                return None
            self._sample_count += 1
            self._last_sample_sec = timestamp
            # Anchor to the scheduled target to avoid accumulating frame rounding.
            self._next_due_sec = target + self._interval
            return FrameSample(
                camera_id=self.camera_id,
                frame_id=f"{self.camera_id}-{self._sample_count:06d}",
                video_timestamp_sec=timestamp,
                captured_at=datetime.now(timezone.utc).isoformat(),
                sampling_interval_sec=self._interval,
                frame_bgr=frame,
            )
        self._finish()
        return None

    def _finish(self) -> None:
        self.exhausted = True
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._capture.release()
            self._closed = True

    def __enter__(self) -> FrameSampler:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def samples(self, *, realtime: bool = True) -> Iterator[FrameSample]:
        """Single-camera convenience iterator; use read_due for multiple cameras."""
        while not self._closed:
            sample = self.read_due(realtime=realtime)
            if sample is not None:
                yield sample
            elif realtime and not self._closed:
                time.sleep(self.seconds_until_due())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="append", required=True, metavar="CAM_01=PATH",
                        help="Repeat for CAM_02 / CAM_03; local video files only")
    parser.add_argument("--interval-sec", type=float, default=DEFAULT_INTERVAL_SEC)
    parser.add_argument("--no-realtime", action="store_true", help="Extract as fast as decoding permits")
    parser.add_argument("--output-dir", type=Path, help="Optionally save sampled JPEGs here")
    args = parser.parse_args()
    try:
        _validate_interval(args.interval_sec)
        sources = {}
        for entry in args.video:
            camera_id, separator, path = entry.partition("=")
            if not separator or camera_id not in CAMERA_IDS or not path:
                raise ValueError("--video must be CAM_01=PATH, CAM_02=PATH or CAM_03=PATH")
            if camera_id in sources:
                raise ValueError(f"Duplicate camera: {camera_id}")
            sources[camera_id] = path
        with ExitStack() as stack:
            samplers = [stack.enter_context(FrameSampler(path, camera, args.interval_sec))
                        for camera, path in sources.items()]
            if args.output_dir:
                args.output_dir.mkdir(parents=True, exist_ok=True)
            while any(not sampler.exhausted for sampler in samplers):
                for sampler in samplers:
                    sample = sampler.read_due(realtime=not args.no_realtime)
                    if sample is None:
                        continue
                    if args.output_dir:
                        output = args.output_dir / f"{sample.frame_id}.jpg"
                        if not cv2.imwrite(str(output), sample.frame_bgr):
                            raise OSError(f"Could not write JPEG: {output}")
                    print(json.dumps(sample.metadata()), flush=True)
                if not args.no_realtime:
                    waits = [s.seconds_until_due() for s in samplers if not s.exhausted]
                    if waits:
                        time.sleep(min(waits))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"frame automation: {exc}\n")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
