import asyncio
import json
import unittest

import httpx

from rsu.decision_engine import CLASS_ORDER, DecisionEngine, normalize_scores
from rsu.frame_sender import FrameSender
from rsu.semantic_sender import SemanticSender
from rsu.vlm import DEFAULT_PROMPTS, parse_prompt_file

NORMAL = (0.85, 0.05, 0.05, 0.05)
CONGESTION = (0.05, 0.85, 0.05, 0.05)
ACCIDENT = (0.05, 0.05, 0.85, 0.05)
INVALID = (0.05, 0.05, 0.05, 0.85)


class DecisionTests(unittest.TestCase):
    def test_prompt_and_score_order(self):
        prompts = parse_prompt_file(DEFAULT_PROMPTS)
        self.assertEqual(tuple(prompts), ("normal_flow", "congestion", "accident", "invalid_scene"))
        self.assertEqual([len(prompts[label]) for label in CLASS_ORDER], [30] * 4)
        scores = normalize_scores([0.1, 0.3, 0.2, 0.1])
        self.assertAlmostEqual(sum(scores), 1)
        self.assertEqual(CLASS_ORDER[scores.index(max(scores))], "congestion")

    def test_verification_and_recovery_are_independent_per_camera(self):
        engine = DecisionEngine()
        self.assertEqual(engine.update("CAM_01", NORMAL).sampling_interval_sec, 2)
        decision = engine.update("CAM_02", CONGESTION)
        self.assertEqual((decision.verification_state, decision.sampling_interval_sec), ("verifying", 1))
        decision = engine.update("CAM_02", CONGESTION)
        self.assertEqual(decision.alert["level"], "confirmed")
        self.assertEqual(engine.update("CAM_01", NORMAL).sampling_interval_sec, 2)
        self.assertEqual(engine.update("CAM_02", NORMAL).event_id, decision.event_id)
        self.assertEqual(engine.update("CAM_02", NORMAL).sampling_interval_sec, 1)
        self.assertEqual(engine.update("CAM_02", NORMAL).sampling_interval_sec, 2)

    def test_accident_keeps_event_id_from_preliminary_to_confirmed(self):
        engine = DecisionEngine()
        preliminary = engine.update("CAM_01", ACCIDENT)
        self.assertEqual(preliminary.alert["level"], "preliminary")
        confirmed = engine.update("CAM_01", ACCIDENT)
        self.assertEqual(confirmed.alert["level"], "confirmed")
        self.assertEqual(confirmed.event_id, preliminary.event_id)

    def test_invalid_frames_have_their_own_state(self):
        engine = DecisionEngine()
        decision = engine.update("CAM_01", INVALID)
        self.assertEqual(decision.verification_state, "invalid")
        self.assertIsNone(decision.alert)


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_semantic_retries_and_reports_delivered_body_size(self):
        requests = []

        def handle(request):
            requests.append(request)
            return httpx.Response(503 if len(requests) == 1 else 204)

        async with httpx.AsyncClient(
            base_url="http://tcc.test", transport=httpx.MockTransport(handle), timeout=2
        ) as client:
            sender = SemanticSender(client, queue_size=1)
            payload = {"camera_id": "CAM_01", "frame_id": "CAM_01-000001"}
            self.assertTrue(sender.submit(payload))
            self.assertFalse(sender.submit(payload))
            worker = asyncio.create_task(sender.run())
            try:
                await asyncio.wait_for(sender.drain(), timeout=2)
            finally:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)

        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[1].url.path, "/api/v1/semantic")
        self.assertEqual(json.loads(requests[1].content), payload)
        self.assertEqual(sender.delivered_bytes, len(requests[1].content))

    async def test_snapshot_queue_drops_stale_preview_and_preserves_frame_id(self):
        requests = []

        def handle(request):
            requests.append(request)
            return httpx.Response(204)

        async with httpx.AsyncClient(
            base_url="http://tcc.test", transport=httpx.MockTransport(handle), timeout=2
        ) as client:
            sender = FrameSender(client, queue_size=1)
            sender.submit(b"old-preview", {"frame_id": "old-frame"})
            sender.submit(b"new-preview", {"frame_id": "new-frame"})
            worker = asyncio.create_task(sender.run())
            try:
                await asyncio.wait_for(sender.drain(), timeout=2)
            finally:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)

        self.assertEqual(sender.dropped_previews, 1)
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request.url.path, "/api/v1/snapshot")
        self.assertIn(b'name="meta"', request.content)
        self.assertIn(b'name="file"', request.content)
        self.assertIn(b"new-frame", request.content)
        self.assertIn(b"new-preview", request.content)
        self.assertNotIn(b"old-preview", request.content)
        self.assertEqual(sender.delivered_jpeg_bytes, len(b"new-preview"))
