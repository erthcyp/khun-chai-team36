"""Mac 2 receiver for RSU semantic updates and asynchronous JPEG previews."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from tcc.meaning_process import MeaningProcessor


CLASS_ORDER = ["normal_flow", "congestion", "accident", "invalid_scene"]
CAMERA_ID = re.compile(r"^CAM_0[1-6]$")
MAX_SEMANTIC_BYTES = 64 * 1024
MAX_JPEG_BYTES = 4 * 1024 * 1024
MATCH_BUFFER_SIZE = 8
SNAPSHOT_HISTORY_SIZE = 4
RUN_ID = re.compile(r"[0-9]{20}-[a-f0-9]{32}")


def checked_run_id(payload: dict) -> str:
    value = payload.get("run_id")
    if value is None:
        return ""
    if not isinstance(value, str) or not RUN_ID.fullmatch(value):
        raise HTTPException(422, "invalid run_id")
    return value


def frame_number(camera_id: str, frame_id: str) -> int:
    match = re.fullmatch(re.escape(camera_id) + r"-(\d+)", frame_id)
    if not match:
        raise HTTPException(422, "frame_id must belong to camera_id and end in a frame number")
    return int(match.group(1))


def checked_camera_id(value: object) -> str:
    if not isinstance(value, str) or not CAMERA_ID.fullmatch(value):
        raise HTTPException(422, "camera_id must be CAM_01 through CAM_06")
    return value


def captured_time(value: object) -> datetime:
    try:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            raise ValueError
        return timestamp.astimezone(timezone.utc)
    except ValueError as exc:
        raise HTTPException(422, "captured_at must be an ISO timestamp with timezone") from exc


def snapshot_matches(metadata: dict, prediction: dict) -> bool:
    timestamp = metadata.get("captured_at")
    event_id = metadata.get("event_id")
    return (not timestamp or captured_time(timestamp) == captured_time(prediction["captured_at"])) and (
        not event_id or event_id == prediction.get("event_id")
    )


class ReceiverState:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.cameras: dict[str, dict] = {}
        self.last_captured: dict[str, datetime] = {}
        self.runs: dict[str, str] = {}
        self.sequences: dict[str, int] = {}
        self.snapshots: dict[str, tuple[tuple[str, int], bytes, dict]] = {}
        self.snapshot_history: dict[str, OrderedDict[str, bytes]] = {}
        self.predictions: dict[str, OrderedDict] = {}
        self.pending_images: dict[str, OrderedDict] = {}
        self.alerts: dict[str, dict] = {}
        self.semantic_count = 0
        self.semantic_bytes = 0
        self.snapshot_count = 0
        self.jpeg_bytes = 0
        self.meaning = MeaningProcessor()
        self.meaning_updates: dict[str, tuple] = {}
        self.meaning_tasks: dict[str, asyncio.Task] = {}
        self.subscribers: set[WebSocket] = set()

    @staticmethod
    def remember(buffer: OrderedDict, key: tuple, value: object) -> None:
        buffer[key] = value
        buffer.move_to_end(key)
        if len(buffer) > MATCH_BUFFER_SIZE:
            buffer.popitem(last=False)

    def publish_snapshot(self, camera_id: str, key: tuple[str, int]) -> None:
        predictions = self.predictions.get(camera_id, {})
        images = self.pending_images.get(camera_id, {})
        if key not in predictions or key not in images:
            return
        jpeg, metadata = images.pop(key)
        prediction = predictions[key]
        previous = self.snapshots.get(camera_id)
        if previous and key <= previous[0]:
            return
        if not snapshot_matches(metadata, prediction):
            return
        version = f"{key[0] or 'legacy'}-{prediction['frame_id']}"
        info = {
            field: prediction.get(field) for field in (
                "frame_id", "captured_at", "predicted_class", "confidence",
                "verification_state", "alert", "video_timestamp_sec", "inference_ms", "received_at",
            )
        }
        info.update(run_id=key[0] or None, version=version, image_received_at=metadata["received_at"])
        history = self.snapshot_history.setdefault(camera_id, OrderedDict())
        history[version] = jpeg
        if len(history) > SNAPSHOT_HISTORY_SIZE:
            history.popitem(last=False)
        self.snapshots[camera_id] = (key, jpeg, info)

    def attach_snapshot(self, camera_id: str) -> None:
        camera = self.cameras[camera_id]
        snapshot = self.snapshots.get(camera_id)
        if snapshot:
            info = snapshot[2]
            camera["snapshot"] = info
            camera["latest_snapshot_url"] = f"/api/v1/snapshots/{camera_id}?v={info['version']}"
        camera["snapshot_pending"] = not snapshot or snapshot[2]["frame_id"] != camera["frame_id"]

    async def broadcast(self, message: dict) -> None:
        for subscriber in tuple(self.subscribers):
            try:
                await asyncio.wait_for(subscriber.send_json(message), timeout=0.2)
            except (RuntimeError, WebSocketDisconnect, asyncio.TimeoutError):
                self.subscribers.discard(subscriber)

    def update_meaning(self, camera_id: str, payload: dict) -> None:
        alert = payload.get("alert")
        signature = (
            payload.get("run_id"), payload["predicted_class"], payload.get("verification_state"),
            alert.get("level") if isinstance(alert, dict) else None,
            payload.get("event_id") or None,
        )
        if self.meaning_updates.get(camera_id) == signature:
            return
        self.meaning_updates[camera_id] = signature
        previous = self.meaning_tasks.get(camera_id)
        if previous and not previous.done():
            previous.cancel()
        self.cameras[camera_id]["meaning_pending"] = True
        self.meaning_tasks[camera_id] = asyncio.create_task(
            self.publish_meaning(camera_id, payload),
        )

    async def publish_meaning(self, camera_id: str, payload: dict) -> None:
        summary = await self.meaning.on_semantic(payload)
        async with self.lock:
            if self.meaning_tasks.get(camera_id) is not asyncio.current_task():
                return
            camera = self.cameras[camera_id]
            camera.pop("meaning_pending", None)
            if summary is not None:
                summary = {**summary, "run_id": payload.get("run_id"),
                           "source_frame_id": payload["frame_id"],
                           "source_captured_at": payload["captured_at"]}
                camera["meaning"] = summary
                if camera_id in self.alerts:
                    self.alerts[camera_id]["meaning"] = summary
        if summary is not None:
            await self.broadcast({"type": "meaning", "data": {"camera_id": camera_id, **summary}})


@asynccontextmanager
async def lifespan(app: FastAPI):
    receiver = ReceiverState()
    app.state.receiver = receiver
    try:
        yield
    finally:
        for task in receiver.meaning_tasks.values():
            task.cancel()
        await asyncio.gather(*receiver.meaning_tasks.values(), return_exceptions=True)


app = FastAPI(title="RoadSense TCC receiver", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4173", "http://127.0.0.1:4173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.mount("/dashboard", StaticFiles(directory=Path(__file__).parent / "dashboard", html=True), name="dashboard")


@app.get("/")
def home() -> RedirectResponse:
    return RedirectResponse("/dashboard/")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "roadsense-tcc", "version": "1.0"}


@app.post("/api/v1/semantic")
async def receive_semantic(request: Request) -> dict:
    body = await request.body()
    if len(body) > MAX_SEMANTIC_BYTES:
        raise HTTPException(413, "semantic message too large")
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(422, "invalid semantic JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(422, "semantic payload must be an object")
    camera_id = checked_camera_id(payload.get("camera_id"))
    frame_id = payload.get("frame_id")
    if not isinstance(frame_id, str):
        raise HTTPException(422, "frame_id is required")
    sequence = frame_number(camera_id, frame_id)
    run = checked_run_id(payload)
    if payload.get("schema_version") != "1.0" or payload.get("message_type") != "semantic_update":
        raise HTTPException(422, "unsupported semantic schema or message type")
    if payload.get("score_order") != CLASS_ORDER or payload.get("predicted_class") not in CLASS_ORDER:
        raise HTTPException(422, "invalid class or score order")
    scores = payload.get("scores")
    if not isinstance(scores, list) or len(scores) != 4 or not all(
        isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score) for score in scores
    ):
        raise HTTPException(422, "scores must contain four numbers")
    timestamp = captured_time(payload.get("captured_at"))
    receiver: ReceiverState = request.app.state.receiver
    async with receiver.lock:
        previous_run = receiver.runs.get(camera_id)
        if previous_run is not None and run < previous_run:
            return {"accepted": False, "reason": "previous_run", "camera_id": camera_id}
        same_run = previous_run == run
        if same_run and (sequence <= receiver.sequences[camera_id]
                         or timestamp < receiver.last_captured[camera_id]):
            return {"accepted": False, "reason": "stale", "camera_id": camera_id}
        receiver.semantic_count += 1
        receiver.semantic_bytes += len(body)
        if not same_run:
            receiver.snapshots.pop(camera_id, None)
            receiver.snapshot_history.pop(camera_id, None)
            receiver.predictions.pop(camera_id, None)
            images = receiver.pending_images.get(camera_id, {})
            receiver.pending_images[camera_id] = OrderedDict(
                (key, value) for key, value in images.items() if key[0] >= run
            )
        receiver.runs[camera_id] = run
        receiver.sequences[camera_id] = sequence
        receiver.last_captured[camera_id] = timestamp
        previous_camera = receiver.cameras.get(camera_id, {}) if same_run else {}
        camera = dict(payload)
        for field in ("meaning", "meaning_pending", "snapshot", "latest_snapshot_url", "snapshot_pending"):
            camera.pop(field, None)
        for field in ("meaning", "meaning_pending"):
            if field in previous_camera:
                camera[field] = previous_camera[field]
        camera["online"] = True
        camera["received_at"] = datetime.now(timezone.utc).isoformat()
        receiver.cameras[camera_id] = camera
        key = (run, sequence)
        receiver.remember(receiver.predictions.setdefault(camera_id, OrderedDict()), key, dict(camera))
        receiver.publish_snapshot(camera_id, key)
        receiver.attach_snapshot(camera_id)
        alert = payload.get("alert")
        if isinstance(alert, dict):
            receiver.alerts[camera_id] = {
                **alert, "camera_id": camera_id, "location": payload.get("location"),
                "captured_at": payload["captured_at"], "frame_id": frame_id,
            }
        elif alert is None:
            receiver.alerts.pop(camera_id, None)
        receiver.update_meaning(camera_id, payload)
        if not camera.get("meaning_pending") and camera_id in receiver.alerts and "meaning" in camera:
            receiver.alerts[camera_id]["meaning"] = camera["meaning"]
    await receiver.broadcast({"type": "semantic", "data": camera})
    return {"accepted": True, "camera_id": camera_id, "frame_id": frame_id}


@app.post("/api/v1/snapshot")
async def receive_snapshot(request: Request, meta: str = Form(...), file: UploadFile = File(...)) -> dict:
    try:
        metadata = json.loads(meta)
    except ValueError as exc:
        raise HTTPException(422, "invalid snapshot metadata") from exc
    if not isinstance(metadata, dict):
        raise HTTPException(422, "snapshot metadata must be an object")
    camera_id = checked_camera_id(metadata.get("camera_id"))
    frame_id = metadata.get("frame_id")
    if not isinstance(frame_id, str):
        raise HTTPException(422, "frame_id is required")
    sequence = frame_number(camera_id, frame_id)
    run = checked_run_id(metadata)
    if run or metadata.get("captured_at") is not None:
        captured_time(metadata.get("captured_at"))
    if file.content_type != "image/jpeg":
        raise HTTPException(415, "snapshot must be image/jpeg")
    jpeg = await file.read(MAX_JPEG_BYTES + 1)
    if len(jpeg) > MAX_JPEG_BYTES:
        raise HTTPException(413, "JPEG too large")
    if not (jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")):
        raise HTTPException(422, "invalid JPEG data")
    receiver: ReceiverState = request.app.state.receiver
    async with receiver.lock:
        current_run = receiver.runs.get(camera_id)
        if current_run is not None and run < current_run:
            return {"accepted": False, "reason": "previous_run", "camera_id": camera_id}
        key = (run, sequence)
        previous = receiver.snapshots.get(camera_id)
        images = receiver.pending_images.setdefault(camera_id, OrderedDict())
        if (previous and key <= previous[0]) or key in images:
            return {"accepted": False, "reason": "stale", "camera_id": camera_id}
        prediction = receiver.predictions.get(camera_id, {}).get(key)
        if prediction and not snapshot_matches(metadata, prediction):
            return {"accepted": False, "reason": "capture_mismatch", "camera_id": camera_id}
        receiver.snapshot_count += 1
        receiver.jpeg_bytes += len(jpeg)
        received_at = datetime.now(timezone.utc).isoformat()
        receiver.remember(images, key, (jpeg, {**metadata, "received_at": received_at}))
        if current_run == run:
            receiver.publish_snapshot(camera_id, key)
            receiver.attach_snapshot(camera_id)
    await receiver.meaning.on_snapshot(metadata, jpeg)
    return {"accepted": True, "camera_id": camera_id, "frame_id": frame_id}


@app.get("/api/v1/snapshots/{camera_id}")
async def get_snapshot(request: Request, camera_id: str, v: str | None = None) -> Response:
    checked_camera_id(camera_id)
    receiver: ReceiverState = request.app.state.receiver
    async with receiver.lock:
        snapshot = receiver.snapshots.get(camera_id)
        if v is not None:
            image = receiver.snapshot_history.get(camera_id, {}).get(v)
        else:
            image = snapshot[1] if snapshot else None
    if image is None:
        raise HTTPException(404, "no snapshot for camera")
    return Response(image, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/v1/cameras")
async def get_cameras(request: Request) -> dict:
    receiver: ReceiverState = request.app.state.receiver
    async with receiver.lock:
        cameras = [dict(camera) for _, camera in sorted(receiver.cameras.items())]
    now = datetime.now(timezone.utc)
    for camera in cameras:
        camera["online"] = (now - captured_time(camera["received_at"])).total_seconds() <= 12
    return {"cameras": cameras}


@app.get("/api/v1/alerts")
async def get_alerts(request: Request) -> dict:
    receiver: ReceiverState = request.app.state.receiver
    async with receiver.lock:
        alerts = [dict(alert) for _, alert in sorted(receiver.alerts.items())]
    return {"alerts": alerts}


@app.get("/api/v1/metrics")
async def get_metrics(request: Request) -> dict:
    receiver: ReceiverState = request.app.state.receiver
    async with receiver.lock:
        return {"received_semantic_count": receiver.semantic_count,
                "semantic_bytes": receiver.semantic_bytes,
                "received_snapshot_count": receiver.snapshot_count,
                "jpeg_bytes": receiver.jpeg_bytes}


@app.websocket("/ws")
async def websocket_updates(websocket: WebSocket) -> None:
    await websocket.accept()
    receiver: ReceiverState = websocket.app.state.receiver
    receiver.subscribers.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        receiver.subscribers.discard(websocket)
