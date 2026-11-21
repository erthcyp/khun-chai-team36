import asyncio
import json
import os
import unittest
from unittest.mock import patch

import httpx

from tcc.meaning_process import MeaningProcessor


FRAME = {
    "camera_id": "CAM_01",
    "frame_id": "CAM_01-000028",
    "prediction": "normal_flow",
    "confidence": 0.6906568139569704,
    "image_path": "assets/frames/CAM_01-000028.jpg",
    "captured_at": "2026-11-21T05:15:47.888826+00:00",
}


def openai_response():
    return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
        "summary_en": "Model predicts normal traffic flow.",
        "summary_th": "โมเดลประเมินว่าการจราจรไหลปกติ",
        "suggested_action": "แนะนำให้ติดตามสถานการณ์ต่อไป",
    })}}]})


class MeaningProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def test_template_accepts_frame_output_and_deduplicates_per_camera(self):
        with patch.dict(os.environ, clear=True):
            processor = MeaningProcessor()

        first = await processor.on_semantic(FRAME)
        self.assertEqual(first["source"], "fallback")
        self.assertEqual(tuple(first)[:2], ("summary_th", "summary_en"))
        self.assertIn("การจราจรไหลปกติ", first["summary_th"])
        self.assertEqual(first["suggested_action"], "แนะนำให้ติดตามสถานการณ์ต่อไป")
        self.assertIn("normal traffic flow", first["summary_en"])
        self.assertIn("0.69", first["summary_en"])
        self.assertNotIn("confirmed", first["summary_en"])
        self.assertIsNone(first["event_id"])
        self.assertIsNone(await processor.on_semantic(dict(FRAME, confidence=0.81)))
        self.assertIsNotNone(await processor.on_semantic(dict(FRAME, camera_id="CAM_02")))
        alert = dict(FRAME, predicted_class="accident", event_id="evt-1",
                     verification_state="alert", alert={"level": "confirmed", "type": "accident"})
        self.assertIn("Confirmed alert", (await processor.on_semantic(alert))["summary_en"])

    async def test_api_trigger_and_snapshot_matching(self):
        calls = []

        async def send(request):
            calls.append(json.loads(request.content))
            return openai_response()

        async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                processor = MeaningProcessor(client=client)
            event = dict(FRAME, event_id="evt-1", verification_state="verifying",
                         location={"label": "Junction A"})
            task = asyncio.create_task(processor.on_semantic(event))
            await asyncio.sleep(0)
            self.assertEqual(len(calls), 0)
            self.assertIsNone(await processor.on_semantic(dict(event, frame_id="later")))
            self.assertIsNone(await processor.on_snapshot(
                dict(camera_id="CAM_01", frame_id=FRAME["frame_id"], event_id="wrong"), b"jpeg"))
            await processor.on_snapshot(
                dict(camera_id="CAM_01", frame_id="other", event_id="evt-1"), b"wrong frame")
            await asyncio.sleep(0)
            meta = dict(camera_id="CAM_01", frame_id=FRAME["frame_id"], event_id="evt-1")
            self.assertEqual(len(calls), 0)
            await processor.on_snapshot(meta, b"jpeg")
            self.assertEqual((await task)["source"], "json+image")
            await processor.on_snapshot(meta, b"jpeg")
            next_event = dict(event, frame_id="next", event_id="evt-2")
            await processor.on_snapshot(
                dict(camera_id="CAM_01", frame_id="next", event_id="evt-2"), b"early")
            self.assertEqual((await processor.on_semantic(next_event))["source"], "json+image")

        self.assertEqual(len(calls), 2)
        self.assertIn("primary operator summary in natural Thai",
                      calls[0]["messages"][0]["content"])
        self.assertIn("โมเดลทำนายจากเฟรมเดียวว่า", calls[0]["messages"][0]["content"])
        self.assertIn("The model predicts from a single frame that",
                      calls[0]["messages"][0]["content"])
        content = calls[0]["messages"][1]["content"]
        facts = json.loads(content[0]["text"])
        self.assertEqual(facts["predicted_class"], "normal_flow")
        self.assertEqual(facts["location"], "Junction A")
        self.assertEqual(facts["captured_at"], FRAME["captured_at"])
        self.assertNotIn("image_path", content[0]["text"])
        self.assertEqual(calls[0]["messages"][1]["content"][1]["type"], "image_url")

    async def test_missing_image_skips_api_and_api_timeout_uses_template(self):
        calls = []

        async def send(request):
            calls.append(request)
            await asyncio.sleep(60)

        async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                processor = MeaningProcessor(client=client, timeout=0.01, image_timeout=0.01)
            self.assertEqual(
                (await processor.on_semantic(FRAME))["fallback_reason"], "image_timeout",
            )
            await processor.on_snapshot(FRAME, b"late")
            self.assertEqual(len(calls), 0)
            camera_two = dict(FRAME, camera_id="CAM_02")
            await processor.on_snapshot(camera_two, b"early")
            self.assertEqual(
                (await processor.on_semantic(camera_two))["fallback_reason"], "api_error",
            )
            self.assertEqual(len(calls), 1)

    async def test_stale_in_flight_summary_is_discarded(self):
        first_started = asyncio.Event()
        release_first = asyncio.Event()

        async def send(request):
            if not first_started.is_set():
                first_started.set()
                await release_first.wait()
            return openai_response()

        async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                processor = MeaningProcessor(client=client)
            await processor.on_snapshot(FRAME, b"first")
            first = asyncio.create_task(processor.on_semantic(FRAME))
            await first_started.wait()
            self.assertIsNone(await processor.on_semantic(FRAME))
            changed = dict(FRAME, frame_id="next", prediction="congestion")
            await processor.on_snapshot(changed, b"second")
            self.assertIsNotNone(await processor.on_semantic(changed))
            release_first.set()
            self.assertIsNone(await first)
