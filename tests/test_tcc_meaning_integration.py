import asyncio
import json
import os
import unittest
from unittest.mock import patch

import httpx

from tcc.meaning_process import MeaningProcessor
from tcc.pc2_receiver import CLASS_ORDER, app


PAYLOAD = {
    "schema_version": "1.0",
    "message_type": "semantic_update",
    "camera_id": "CAM_01",
    "frame_id": "CAM_01-000001",
    "captured_at": "2026-11-21T05:15:47+00:00",
    "predicted_class": "congestion",
    "score_order": CLASS_ORDER,
    "scores": [0.1, 0.8, 0.06, 0.04],
    "confidence": 0.8,
    "verification_state": "verifying",
    "location": {"label": "Junction A"},
    "event_id": "event-1",
    "alert": {"level": "preliminary", "type": "congestion"},
}


class ReceiverMeaningTests(unittest.IsolatedAsyncioTestCase):
    async def test_routes_publish_camera_before_bilingual_meaning_and_keep_summary(self):
        calls = []

        async def openai(request):
            calls.append(json.loads(request.content))
            data = {"summary_th": "การจราจรติดขัด", "summary_en": "Congestion",
                    "suggested_action": "ตรวจสอบภาพ"}
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(data)}}]})

        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.MockTransport(openai)) as model,
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as api,
        ):
            receiver = app.state.receiver
            with patch.dict(os.environ, {"roadsense_meaning": "test-key"}):
                receiver.meaning = MeaningProcessor(client=model)

            reply = await api.post("/api/v1/semantic", json=PAYLOAD)
            self.assertTrue(reply.json()["accepted"])
            camera = (await api.get("/api/v1/cameras")).json()["cameras"][0]
            self.assertTrue(camera["meaning_pending"])
            self.assertEqual(calls, [])

            snapshot = {"camera_id": "CAM_01", "frame_id": PAYLOAD["frame_id"], "event_id": "event-1"}
            reply = await api.post("/api/v1/snapshot", data={"meta": json.dumps(snapshot)},
                                   files={"file": ("frame.jpg", b"\xff\xd8demo\xff\xd9", "image/jpeg")})
            self.assertTrue(reply.json()["accepted"])
            await asyncio.wait_for(receiver.meaning_tasks["CAM_01"], 1)
            camera = (await api.get("/api/v1/cameras")).json()["cameras"][0]
            alert = (await api.get("/api/v1/alerts")).json()["alerts"][0]
            summary = camera["meaning"]
            self.assertEqual(summary, alert["meaning"])
            self.assertEqual(summary["summary_th"], "การจราจรติดขัด")
            self.assertEqual(summary["summary_en"], "Congestion")
            self.assertEqual(summary["source"], "json+image")

            reply = await api.post("/api/v1/semantic", json=dict(PAYLOAD, frame_id="CAM_01-000002"))
            self.assertTrue(reply.json()["accepted"])
            camera = (await api.get("/api/v1/cameras")).json()["cameras"][0]
            self.assertEqual(camera["meaning"], summary)
            self.assertEqual(len(calls), 1)
