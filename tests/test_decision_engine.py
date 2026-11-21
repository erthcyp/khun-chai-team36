import unittest

from rsu.decision_engine import DecisionConfig, DecisionEngine

NORMAL = (0.85, 0.05, 0.05, 0.05)
CONGESTION = (0.05, 0.85, 0.05, 0.05)
ACCIDENT = (0.05, 0.05, 0.85, 0.05)
INVALID = (0.05, 0.05, 0.05, 0.85)
UNCERTAIN_NORMAL = (0.40, 0.35, 0.20, 0.05)


class DecisionEngineTests(unittest.TestCase):
    def test_accident_preliminary_alert_starts_at_high_confidence_threshold(self):
        for scores, initial_level, initial_priority in (
            ((0.20, 0.20, 0.59, 0.01), None, "medium"),
            ((0.20, 0.19, 0.60, 0.01), "preliminary", "critical"),
        ):
            with self.subTest(confidence=scores[2]):
                engine = DecisionEngine()
                first = engine.update("CAM_01", scores)
                self.assertEqual(first.verification_state, "verifying")
                self.assertEqual(first.sampling_interval_sec, 1.0)
                self.assertEqual((first.alert or {}).get("level"), initial_level)
                self.assertEqual(first.priority, initial_priority)
                second = engine.update("CAM_01", scores)
                self.assertEqual(second.verification_state, "alert")
                self.assertEqual(second.alert["level"], "confirmed")

    def test_interrupted_congestion_needs_two_fresh_reliable_frames(self):
        for interruption in (NORMAL, INVALID, (0.35, 0.40, 0.20, 0.05)):
            with self.subTest(interruption=interruption):
                engine = DecisionEngine()
                self.assertIsNone(engine.update("CAM_01", CONGESTION).alert)
                self.assertIsNone(engine.update("CAM_01", interruption).alert)
                first = engine.update("CAM_01", CONGESTION)
                self.assertEqual(first.verification_state, "verifying")
                self.assertIsNone(first.alert)
                confirmed = engine.update("CAM_01", CONGESTION)
                self.assertEqual(confirmed.alert["level"], "confirmed")
                self.assertEqual(confirmed.priority, "high")

    def test_uncertain_normal_frame_restarts_recovery_and_event_clears_after_three(self):
        engine = DecisionEngine()
        engine.update("CAM_01", CONGESTION)
        confirmed = engine.update("CAM_01", CONGESTION)
        for _ in range(2):
            self.assertEqual(engine.update("CAM_01", NORMAL).verification_state, "recovery")

        uncertain = engine.update("CAM_01", UNCERTAIN_NORMAL)
        self.assertEqual(uncertain.verification_state, "verifying")
        for _ in range(2):
            recovering = engine.update("CAM_01", NORMAL)
            self.assertEqual(recovering.verification_state, "recovery")
            self.assertEqual(recovering.sampling_interval_sec, 1.0)
            self.assertEqual(recovering.event_id, confirmed.event_id)

        recovered = engine.update("CAM_01", NORMAL)
        self.assertEqual(recovered.verification_state, "normal")
        self.assertEqual(recovered.sampling_interval_sec, 2.0)
        self.assertEqual(recovered.priority, "low")
        self.assertIsNone(recovered.event_id)
        self.assertIsNone(recovered.alert)

    def test_two_cameras_count_incident_frames_separately(self):
        engine = DecisionEngine(DecisionConfig(confirm_frames=3))
        for _ in range(2):
            for camera_id in ("CAM_01", "CAM_02"):
                self.assertIsNone(engine.update(camera_id, CONGESTION).alert)

        cam_01 = engine.update("CAM_01", CONGESTION)
        cam_02 = engine.update("CAM_02", CONGESTION)
        self.assertEqual(cam_01.alert["level"], "confirmed")
        self.assertEqual(cam_02.alert["level"], "confirmed")
        self.assertNotEqual(cam_01.event_id, cam_02.event_id)
        self.assertEqual(engine.update("CAM_01", CONGESTION).event_id, cam_01.event_id)
        self.assertEqual(engine.update("CAM_02", CONGESTION).event_id, cam_02.event_id)

    def test_pending_review_interrupts_accident_confirmation(self):
        engine = DecisionEngine(DecisionConfig(vision_review_margin=0.30))
        preliminary = engine.update("CAM_01", ACCIDENT, defer_close_scores=True)
        pending = engine.update("CAM_01", (0.35, 0.02, 0.62, 0.01), defer_close_scores=True)
        self.assertTrue(pending.review_required)
        self.assertEqual(pending.verification_state, "verifying")
        self.assertEqual(pending.alert["level"], "preliminary")
        self.assertEqual(pending.event_id, preliminary.event_id)

        first = engine.update("CAM_01", ACCIDENT, defer_close_scores=True)
        self.assertFalse(first.review_required)
        self.assertEqual(first.alert["level"], "preliminary")
        confirmed = engine.update("CAM_01", ACCIDENT, defer_close_scores=True)
        self.assertEqual(confirmed.alert["level"], "confirmed")
        self.assertEqual(confirmed.event_id, preliminary.event_id)

    def test_invalid_score_vectors_leave_confirmation_progress_intact(self):
        engine = DecisionEngine()
        self.assertIsNone(engine.update("CAM_01", CONGESTION).alert)
        for scores in (
            (0.5, 0.5), (-0.1, 0.5, 0.4, 0.2), (1.1, 0, 0, 0), (float("nan"), 0, 0, 0),
        ):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                engine.update("CAM_01", scores)
        self.assertEqual(engine.update("CAM_01", CONGESTION).alert["level"], "confirmed")
