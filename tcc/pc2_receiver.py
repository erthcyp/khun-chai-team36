"""Mac 2 receiver for RSU semantic updates and asynchronous JPEG previews."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from tcc.meaning_process import MeaningProcessor


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
        self.semantic_count = 0
        self.semantic_bytes = 0
        self.snapshot_count = 0
        self.jpeg_bytes = 0
        self.meaning = MeaningProcessor()
        self.meaning_updates: dict[str, tuple] = {}
        self.meaning_tasks: dict[str, asyncio.Task] = {}

    def update_meaning(self, camera_id: str, payload: dict) -> None:
        alert = payload.get("alert")
        signature = (
            payload["predicted_class"], payload.get("verification_state"),
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
                camera["meaning"] = summary
                if camera_id in self.alerts:
                    self.alerts[camera_id]["meaning"] = summary


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
        previous_camera = receiver.cameras.get(camera_id, {})
        camera = dict(payload)
        for field in ("meaning", "meaning_pending"):
            camera.pop(field, None)
            if field in previous_camera:
                camera[field] = previous_camera[field]
        camera["online"] = True
        camera["received_at"] = datetime.now(timezone.utc).isoformat()
        if camera_id in receiver.snapshots:
            camera["latest_snapshot_url"] = f"/api/v1/snapshots/{camera_id}"
        receiver.cameras[camera_id] = camera
        alert = payload.get("alert")
        if isinstance(alert, dict):
            receiver.alerts[camera_id] = {
                **alert, "camera_id": camera_id, "location": payload.get("location"),
                "captured_at": payload["captured_at"], "frame_id": frame_id,
            }
        elif alert is None:
            receiver.alerts.pop(camera_id, None)
        receiver.update_meaning(camera_id, payload)
        if camera.get("meaning_pending"):
            camera.pop("meaning", None)
        elif camera_id in receiver.alerts and "meaning" in camera:
            receiver.alerts[camera_id]["meaning"] = camera["meaning"]
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
