"""Bridge the synchronous RSU runner to the asynchronous TCC senders."""

import asyncio
import json
import logging
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread

import cv2
import httpx
import numpy as np

from .frame_sender import FrameSender
from .semantic_sender import SemanticSender

LOG = logging.getLogger(__name__)
FRAME_FIELDS = ("camera_id", "frame_id", "video_timestamp_sec", "captured_at")
RESULT_FIELDS = ("scores", "score_order", "confidence", "inference_ms",
                 "verification_state", "priority", "alert", "event_id")


def load_locations(path: Path, camera_ids: tuple[str, ...]) -> dict[str, dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("camera config must contain a list of cameras")
    locations = {}
    for record in records:
        if not isinstance(record, dict) or record.get("camera_id") not in camera_ids:
            continue
        camera_id = record["camera_id"]
        location = record.get("location")
        if camera_id in locations:
            raise ValueError(f"duplicate camera config: {camera_id}")
        if not isinstance(location, dict) or not isinstance(location.get("label"), str) \
                or not location["label"].strip():
            raise ValueError(f"camera config requires a location label for {camera_id}")
        coordinates = {}
        for field, limit in (("latitude", 90), ("longitude", 180)):
            value = location.get(field)
            if value is not None and (
                type(value) not in (int, float) or not -limit <= value <= limit
            ):
                raise ValueError(f"invalid {field} for {camera_id}")
            coordinates[field] = value
        locations[camera_id] = {"label": location["label"], **coordinates}
    missing = [camera for camera in camera_ids if camera not in locations]
    if missing:
        raise ValueError(f"missing camera config for {', '.join(missing)}")
    return locations


def _encode_preview(frame: np.ndarray) -> bytes | None:
    height, width = frame.shape[:2]
    scale = min(1.0, 960 / max(width, height))
    if scale < 1:
        frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))),
                           interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return encoded.tobytes() if ok else None


class SenderConnector:
    """Own sender workers and bound the synchronous-to-async handoff."""

    def __init__(self, tcc_url: str, locations: dict[str, dict], *,
                 transport: httpx.AsyncBaseTransport | None = None):
        try:
            url = httpx.URL(tcc_url)
        except httpx.InvalidURL as exc:
            raise ValueError("TCC URL must be an HTTP(S) origin") from exc
        if url.scheme not in ("http", "https") or not url.host or url.path != "/" \
                or url.query or url.fragment or url.userinfo:
            raise ValueError("TCC URL must be an HTTP(S) origin")
        self._url = url
        self._locations = locations
        self._transport = transport
        self._queue: Queue[tuple[dict, np.ndarray]] = Queue(maxsize=8)
        self._closing = Event()
        self._ready = Event()
        self._worker = Thread(target=self._run_worker, name="roadsense-tcc-senders")
        self._handoff_drops = 0
        self._semantic_drops = 0

    def __enter__(self):
        self._worker.start()
        self._ready.wait()
        return self

    def __exit__(self, *_exc):
        self._closing.set()
        if self._worker.is_alive():
            self._notify()
        self._worker.join()
        if self._handoff_drops or self._semantic_drops:
            LOG.warning("Dropped RSU updates: handoff=%d, semantic=%d",
                        self._handoff_drops, self._semantic_drops)

    def _notify(self) -> None:
        try:
            self._loop.call_soon_threadsafe(self._wakeup.set)
        except RuntimeError:
            # The network worker already shut down.
            pass

    def submit(self, record: dict, frame: np.ndarray) -> None:
        """Called by the prediction worker. Frames are immutable after capture."""
        try:
            self._queue.put_nowait((record, frame))
        except Full:
            self._handoff_drops += 1
        else:
            self._notify()

    def _run_worker(self) -> None:
        try:
            asyncio.run(self._pump())
        except Exception:
            LOG.exception("TCC sender worker stopped")
        finally:
            self._ready.set()

    async def _pump(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._wakeup = asyncio.Event()
        async with httpx.AsyncClient(base_url=self._url, timeout=1.0,
                                    transport=self._transport) as client:
            semantic = SemanticSender(client)
            frames = FrameSender(client)
            tasks = [asyncio.create_task(sender.run()) for sender in (semantic, frames)]
            self._ready.set()
            try:
                while True:
                    self._wakeup.clear()
                    try:
                        record, frame = self._queue.get_nowait()
                    except Empty:
                        if self._closing.is_set():
                            break
                        await self._wakeup.wait()
                        continue
                    payload = {field: record[field] for field in (*FRAME_FIELDS, *RESULT_FIELDS)}
                    payload.update(
                        schema_version="1.0",
                        message_type="semantic_update",
                        predicted_class=record["prediction"],
                        sampling_interval_sec=record["next_sampling_interval_sec"],
                        location=self._locations[record["camera_id"]],
                    )
                    if not semantic.submit(payload):
                        self._semantic_drops += 1
                        continue
                    try:
                        jpeg = await asyncio.to_thread(_encode_preview, frame)
                    except cv2.error:
                        LOG.warning("Could not encode snapshot for %s", record["frame_id"])
                        continue
                    if jpeg:
                        metadata = {field: payload[field] for field in (
                            "camera_id", "frame_id", "video_timestamp_sec", "event_id",
                        )}
                        frames.submit(jpeg, metadata)
                try:
                    async with asyncio.timeout(4):
                        await asyncio.gather(semantic.drain(), frames.drain())
                except TimeoutError:
                    LOG.warning("Timed out draining queued TCC updates")
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
