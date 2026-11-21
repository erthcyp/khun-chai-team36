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
        "suggested_action": "Continue monitoring.",
    })}}]})


class MeaningProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def test_template_accepts_frame_output_and_deduplicates_per_camera(self):
        with patch.dict(os.environ, clear=True):
            processor = MeaningProcessor()

        first = await processor.on_semantic(FRAME)
        self.assertEqual(first["source"], "fallback")
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
            event = dict(FRAME, event_id="evt-1", verification_state="verifying")
            self.assertEqual((await processor.on_semantic(event))["source"], "json")
            self.assertIsNone(await processor.on_semantic(dict(event, frame_id="later")))
            self.assertIsNone(await processor.on_snapshot(
                dict(camera_id="CAM_01", frame_id=FRAME["frame_id"], event_id="wrong"), b"jpeg"))
            meta = dict(camera_id="CAM_01", frame_id=FRAME["frame_id"], event_id="evt-1")
            self.assertEqual((await processor.on_snapshot(meta, b"jpeg"))["source"], "json+image")
            self.assertIsNone(await processor.on_snapshot(meta, b"jpeg"))
            self.assertIsNotNone(await processor.on_semantic(dict(event, event_id="evt-2")))
            self.assertIsNotNone(await processor.on_semantic(dict(
                event, event_id="evt-2", alert={"level": "preliminary"})))

        self.assertEqual(len(calls), 4)
        content = calls[0]["messages"][1]["content"]
        self.assertEqual(json.loads(content[0]["text"])["predicted_class"], "normal_flow")
        self.assertNotIn("image_path", content[0]["text"])
        self.assertEqual(calls[1]["messages"][1]["content"][1]["type"], "image_url")

    async def test_timeout_and_bad_response_use_template(self):
        async def send(request):
            if json.loads(request.content)["messages"][1]["content"][0]["text"].find("CAM_01") >= 0:
                await asyncio.sleep(60)
            return httpx.Response(200, json={"choices": []})

        async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                processor = MeaningProcessor(client=client, timeout=0.01)
            self.assertEqual((await processor.on_semantic(FRAME))["source"], "fallback")
            self.assertEqual((await processor.on_semantic(
                dict(FRAME, camera_id="CAM_02")))["source"], "fallback")

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
            first = asyncio.create_task(processor.on_semantic(FRAME))
            await first_started.wait()
            self.assertIsNone(await processor.on_semantic(FRAME))
            self.assertIsNotNone(await processor.on_semantic(dict(FRAME, prediction="congestion")))
            release_first.set()
            self.assertIsNone(await first)
