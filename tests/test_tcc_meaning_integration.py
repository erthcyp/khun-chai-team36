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

    async def test_matches_exact_picture_and_rejects_previous_run_after_restart(self):
        first_run = "00000000000000000001-" + "a" * 32
        second_run = "00000000000000000002-" + "b" * 32
        base = dict(PAYLOAD, run_id=first_run, frame_id="CAM_01-000010")

        async def photo(api, record, content):
            meta = {key: record[key] for key in ("camera_id", "frame_id", "run_id", "captured_at", "event_id")}
            return await api.post("/api/v1/snapshot", data={"meta": json.dumps(meta)},
                                  files={"file": ("frame.jpg", b"\xff\xd8" + content + b"\xff\xd9", "image/jpeg")})

        async def camera(api):
            return (await api.get("/api/v1/cameras")).json()["cameras"][0]

        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as api,
        ):
            self.assertTrue((await api.post("/api/v1/semantic", json=base)).json()["accepted"])
            self.assertTrue((await photo(api, base, b"old")).json()["accepted"])
            old = await camera(api)
            self.assertEqual(old["snapshot"]["predicted_class"], "congestion")
            self.assertFalse(old["snapshot_pending"])

            next_frame = dict(base, frame_id="CAM_01-000011", predicted_class="normal_flow")
            await api.post("/api/v1/semantic", json=next_frame)
            waiting = await camera(api)
            self.assertTrue(waiting["snapshot_pending"])
            self.assertEqual(waiting["predicted_class"], "normal_flow")
            self.assertEqual(waiting["snapshot"]["predicted_class"], "congestion")

            restarted = dict(base, run_id=second_run, frame_id="CAM_01-000001")
            self.assertTrue((await photo(api, restarted, b"new")).json()["accepted"])
            self.assertEqual((await camera(api))["latest_snapshot_url"], old["latest_snapshot_url"])
            self.assertTrue((await api.post("/api/v1/semantic", json=restarted)).json()["accepted"])
            current = await camera(api)
            self.assertFalse(current["snapshot_pending"])
            self.assertEqual(current["snapshot"]["frame_id"], "CAM_01-000001")
            self.assertEqual(current["snapshot"]["run_id"], second_run)
            self.assertIn(second_run, current["latest_snapshot_url"])
            self.assertEqual((await api.get(current["latest_snapshot_url"])).content, b"\xff\xd8new\xff\xd9")
            self.assertEqual((await api.get(old["latest_snapshot_url"])).status_code, 404)
            self.assertEqual((await photo(api, next_frame, b"delayed")).json()["reason"], "previous_run")
            self.assertEqual((await api.post("/api/v1/semantic", json=next_frame)).json()["reason"], "previous_run")
