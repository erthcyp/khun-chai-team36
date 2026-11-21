import base64
import json
import unittest

import cv2
import httpx
import numpy as np

from rsu.decision_engine import DecisionConfig, DecisionEngine
from rsu.frame_automation import FrameSample
from rsu.vision_review import Evidence, OpenAIFrameReviewer, TemporalReview, Verdict


def sample(camera, sequence, seconds):
    return FrameSample(
        camera, f"{camera}-{sequence:06d}", seconds, "2026-11-21T10:30:00+07:00",
        1.0, np.zeros((48, 64, 3), dtype=np.uint8),
    )


class VisionReviewTests(unittest.TestCase):
    def test_only_close_top_two_scores_trigger_review(self):
        engine = DecisionEngine()
        self.assertTrue(engine.update("CAM_01", [0.49, 0.02, 0.48, 0.01],
                                      defer_close_scores=True).review_required)
        self.assertTrue(engine.update("CAM_01", [0.45, 0.10, 0.37, 0.08],
                                      defer_close_scores=True).review_required)
        self.assertFalse(engine.update("CAM_01", [0.52, 0.12, 0.34, 0.02],
                                       defer_close_scores=True).review_required)

    def test_pending_review_does_not_originate_or_confirm_an_accident(self):
        engine = DecisionEngine(DecisionConfig(vision_review_margin=0.30))
        for _ in range(3):
            decision = engine.update("CAM_01", [0.35, 0.02, 0.62, 0.01], defer_close_scores=True)
            self.assertTrue(decision.review_required)
            self.assertEqual(decision.verification_state, "verifying")
            self.assertIsNone(decision.alert)

    def test_request_labels_three_frames_by_time_and_sends_the_crop(self):
        def respond(request):
            self.assertEqual(request.url.path, "/v1/chat/completions")
            body = json.loads(request.content)
            self.assertFalse(body["store"])
            parts = body["messages"][1]["content"]
            self.assertEqual([part["text"] for part in parts[::2]], [
                "PREVIOUS, video time 1.000s", "TARGET, video time 2.000s", "FOLLOWING, video time 3.000s",
            ])
            for part in parts[1::2]:
                encoded = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
                image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(image.shape, (24, 32, 3))
            return httpx.Response(200, json={"choices": [{"message": {
                "content": json.dumps({"label": "accident", "reason": "Visible impact at 2s"})
            }}]})

        reviewer = OpenAIFrameReviewer(
            "test", roi=(0.25, 0.25, 0.5, 0.5), transport=httpx.MockTransport(respond)
        )
        result = reviewer(Evidence(sample("CAM_01", 2, 2), sample("CAM_01", 1, 1), sample("CAM_01", 3, 3)))
        self.assertEqual(result, Verdict("accident", "Visible impact at 2s"))

    def test_unavailable_or_uncertain_remote_verdict_requires_human_review(self):
        for reply in (
            httpx.Response(429),
            httpx.Response(200, json={"choices": [{"message": {"content": "invalid"}}]}),
            httpx.Response(200, json={"choices": [{"message": {
                "content": '{"label":"uncertain","reason":"View is blocked"}'
            }}]}),
        ):
            with self.subTest(status=reply.status_code):
                reviewer = OpenAIFrameReviewer("test", transport=httpx.MockTransport(lambda _: reply))
                self.assertIsNone(reviewer(Evidence(sample("CAM_01", 1, 0))).label)

    def test_camera_context_is_independent_and_eof_uses_available_frames(self):
        review_queue = TemporalReview(lambda _: Verdict("normal_flow", "No stopped vehicles."), clock=lambda: 0.0)
        cam1_previous = sample("CAM_01", 1, 0)
        cam1_target = sample("CAM_01", 2, 2)
        cam2_target = sample("CAM_02", 1, 0)
        cam1_following = sample("CAM_01", 3, 3)
        self.assertIsNone(review_queue.observe(cam1_previous, False))
        self.assertEqual(review_queue.observe(cam1_target, True), "pending")
        self.assertEqual(review_queue.observe(cam2_target, True), "pending")
        self.assertEqual(review_queue.observe(cam1_following, True), "not_queued")
        evidence_by_camera = {
            evidence.target.camera_id: evidence for evidence, _ in review_queue.finish(wait=True)
        }
        self.assertIs(evidence_by_camera["CAM_01"].previous, cam1_previous)
        self.assertIs(evidence_by_camera["CAM_01"].following, cam1_following)
        self.assertIsNone(evidence_by_camera["CAM_02"].previous)
        self.assertIsNone(evidence_by_camera["CAM_02"].following)
