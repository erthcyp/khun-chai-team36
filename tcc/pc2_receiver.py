"""Mac 2 receiver for RSU semantic updates and asynchronous JPEG previews."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from .meaning_process import MeaningProcessor


CLASS_ORDER = ["normal_flow", "congestion", "accident", "invalid_scene"]
CAMERA_ID = re.compile(r"^CAM_0[1-6]$")
MAX_SEMANTIC_BYTES = 64 * 1024
MAX_JPEG_BYTES = 4 * 1024 * 1024


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


class ReceiverState:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.cameras: dict[str, dict] = {}
        self.last_captured: dict[str, datetime] = {}
        self.snapshots: dict[str, tuple[int, bytes]] = {}
        self.alerts: dict[str, dict] = {}
        self.meaning_keys: dict[str, tuple] = {}
        self.meaning = MeaningProcessor()
        self.meaning_tasks: set[asyncio.Task] = set()
        self.subscribers: set[WebSocket] = set()
        self.semantic_count = 0
        self.semantic_bytes = 0
        self.snapshot_count = 0
        self.jpeg_bytes = 0


app = FastAPI(title="RoadSense TCC receiver", version="1.0")
app.state.receiver = ReceiverState()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4173", "http://127.0.0.1:4173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.mount("/dashboard", StaticFiles(directory=Path(__file__).parent / "dashboard", html=True), name="dashboard")


def meaning_key(payload: dict) -> tuple:
    alert = payload.get("alert")
    level = alert.get("level") if isinstance(alert, dict) else None
    return (payload.get("predicted_class"), payload.get("verification_state"),
            level, payload.get("event_id"))


async def broadcast(receiver: ReceiverState, message: dict) -> None:
    for subscriber in tuple(receiver.subscribers):
        try:
            await subscriber.send_json(message)
        except (RuntimeError, WebSocketDisconnect):
            receiver.subscribers.discard(subscriber)


async def publish_meaning(receiver: ReceiverState, payload: dict, key: tuple) -> None:
    result = await receiver.meaning.on_semantic(payload)
    if result is None:
        return
    camera_id = payload["camera_id"]
    async with receiver.lock:
        if receiver.meaning_keys.get(camera_id) != key:
            return
        receiver.cameras[camera_id].update(result)
        if camera_id in receiver.alerts:
            receiver.alerts[camera_id].update(result)
    await broadcast(receiver, {"type": "meaning", "data": {"camera_id": camera_id, **result}})


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
    frame_number(camera_id, frame_id)
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
        receiver.semantic_count += 1
        receiver.semantic_bytes += len(body)
        if timestamp < receiver.last_captured.get(camera_id, datetime.min.replace(tzinfo=timezone.utc)):
            return {"accepted": False, "reason": "stale", "camera_id": camera_id}
        receiver.last_captured[camera_id] = timestamp
        key = meaning_key(payload)
        previous = receiver.cameras.get(camera_id, {})
        camera = dict(payload)
        camera["online"] = True
        camera["received_at"] = datetime.now(timezone.utc).isoformat()
        if receiver.meaning_keys.get(camera_id) == key:
            for field in ("summary_th", "summary_en", "suggested_action", "source",
                          "model", "generated_at", "fallback_reason"):
                if field in previous:
                    camera[field] = previous[field]
        if camera_id in receiver.snapshots:
            camera["latest_snapshot_url"] = f"/api/v1/snapshots/{camera_id}"
        receiver.cameras[camera_id] = camera
        receiver.meaning_keys[camera_id] = key
        alert = payload.get("alert")
        if isinstance(alert, dict):
            receiver.alerts[camera_id] = {
                **alert, "camera_id": camera_id, "location": payload.get("location"),
                "captured_at": payload["captured_at"], "frame_id": frame_id,
            }
            for field in ("summary_th", "summary_en", "suggested_action", "source",
                          "model", "generated_at", "fallback_reason"):
                if field in camera:
                    receiver.alerts[camera_id][field] = camera[field]
        elif alert is None:
            receiver.alerts.pop(camera_id, None)
    await broadcast(receiver, {"type": "semantic", "data": camera})
    task = asyncio.create_task(publish_meaning(receiver, payload, key))
    receiver.meaning_tasks.add(task)
    task.add_done_callback(receiver.meaning_tasks.discard)
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
    if file.content_type != "image/jpeg":
        raise HTTPException(415, "snapshot must be image/jpeg")
    jpeg = await file.read(MAX_JPEG_BYTES + 1)
    if len(jpeg) > MAX_JPEG_BYTES:
        raise HTTPException(413, "JPEG too large")
    if not (jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")):
        raise HTTPException(422, "invalid JPEG data")
    receiver: ReceiverState = request.app.state.receiver
    async with receiver.lock:
        receiver.snapshot_count += 1
        receiver.jpeg_bytes += len(jpeg)
        previous = receiver.snapshots.get(camera_id)
        if previous and sequence < previous[0]:
            return {"accepted": False, "reason": "stale", "camera_id": camera_id}
        receiver.snapshots[camera_id] = (sequence, jpeg)
        if camera_id in receiver.cameras:
            receiver.cameras[camera_id]["latest_snapshot_url"] = f"/api/v1/snapshots/{camera_id}"
    await receiver.meaning.on_snapshot(metadata, jpeg)
    return {"accepted": True, "camera_id": camera_id, "frame_id": frame_id}


@app.get("/api/v1/snapshots/{camera_id}")
async def get_snapshot(request: Request, camera_id: str) -> Response:
    checked_camera_id(camera_id)
    receiver: ReceiverState = request.app.state.receiver
    async with receiver.lock:
        snapshot = receiver.snapshots.get(camera_id)
    if snapshot is None:
        raise HTTPException(404, "no snapshot for camera")
    return Response(snapshot[1], media_type="image/jpeg", headers={"Cache-Control": "no-store"})


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
        receiver.subscribers.discard(websocket)
