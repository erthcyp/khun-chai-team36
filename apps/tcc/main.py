"""TCC receiver: ingest 16-byte semantic vectors, run the response agent, serve the dashboard.

Run: uvicorn apps.tcc.main:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from apps.tcc.agent import plan
from semcom.payload import CLASSES, decode, top_class

WEB = Path(__file__).resolve().parents[1] / "web"
ACT_ON = {"accident", "congestion", "invalid_scene"}

app = FastAPI(title="TCC")
events: deque[dict] = deque(maxlen=500)
rsus: dict[str, dict] = {}
clients: set[WebSocket] = set()


async def broadcast(msg: dict):
    for ws in list(clients):
        try:
            await ws.send_json(msg)
        except Exception:  # noqa: BLE001
            clients.discard(ws)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(request: Request):
    body = await request.body()
    try:
        m = decode(body)
    except ValueError as e:
        raise HTTPException(400, str(e))
    h = request.headers
    rsu_id = h.get("x-rsu-id", "unknown")
    state, conf = top_class(m)
    send_ms = int(h.get("x-send-ms", "0") or 0)
    event = {
        "rsu_id": rsu_id,
        "seq": int(h.get("x-seq", "0") or 0),
        "state": state,
        "confidence": round(conf, 4),
        "vector": {c: round(float(x), 4) for c, x in zip(CLASSES, m)},
        "cpv": h.get("x-cpv", "skip"),
        "frame": h.get("x-frame"),
        "payload_bytes": len(body),
        "recv_ms": int(time.time() * 1000),
        "clock_delta_ms": (int(time.time() * 1000) - send_ms) if send_ms else None,
    }
    prev = rsus.get(rsu_id, {}).get("state")
    rsus[rsu_id] = {"state": state, "confidence": event["confidence"], "last_seen": event["recv_ms"]}
    events.append(event)
    await broadcast({"type": "event", "data": event})

    # Only call the agent when the state changes into something actionable.
    if state in ACT_ON and state != prev:
        history = [e for e in events if e["rsu_id"] == rsu_id]
        asyncio.create_task(_run_agent(event, history))
    return Response(status_code=204)


async def _run_agent(event: dict, history: list[dict]):
    result = await asyncio.to_thread(plan, event, history)
    event["plan"] = result
    await broadcast({"type": "plan", "data": {"rsu_id": event["rsu_id"], "seq": event["seq"], "plan": result}})


@app.get("/api/events")
def get_events(limit: int = 50):
    return list(events)[-limit:]


@app.get("/api/rsus")
def get_rsus():
    return rsus


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        clients.discard(websocket)


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/static", StaticFiles(directory=WEB), name="static")
