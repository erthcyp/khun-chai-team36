"""Exercise real OpenCV decoding, using generated local fixtures rather than labels."""

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from rsu.frame_automation import FrameSampler


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class FrameAutomationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.video = Path(cls.temp.name) / "fixture.avi"
        writer = cv2.VideoWriter(str(cls.video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
        if not writer.isOpened():
            raise RuntimeError("OpenCV MJPG encoder is needed for video integration tests")
        try:
            for index in range(51):
                writer.write(np.full((48, 64, 3), index * 4, dtype=np.uint8))
        finally:
            writer.release()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_default_extracts_every_video_second_and_preserves_pixels(self):
        with FrameSampler(self.video) as sampler:
            samples = list(sampler.samples(realtime=False))
            self.assertTrue(sampler.exhausted)
        self.assertEqual([round(s.video_timestamp_sec, 6) for s in samples], [0, 1, 2, 3, 4, 5])
        self.assertEqual([s.frame_id for s in samples], [f"CAM_01-{i:06d}" for i in range(1, 7)])
        for second, sample in enumerate(samples):
            self.assertEqual(sample.frame_bgr.shape, (48, 64, 3))
            self.assertAlmostEqual(float(sample.frame_bgr.mean()), second * 40, delta=2)
            self.assertEqual(sample.sampling_interval_sec, 1.0)
            self.assertIsNotNone(datetime.fromisoformat(sample.captured_at).tzinfo)
            self.assertNotIn("frame_bgr", sample.metadata())

    def test_realtime_does_not_emit_before_due(self):
        clock = FakeClock()
        with FrameSampler(self.video, clock=clock) as sampler:
            self.assertAlmostEqual(sampler.read_due().video_timestamp_sec, 0)
            self.assertIsNone(sampler.read_due())
            clock.now += 0.9
            self.assertIsNone(sampler.read_due())
            self.assertAlmostEqual(sampler.seconds_until_due(), 0.1)
            clock.now += 0.1
            self.assertAlmostEqual(sampler.read_due().video_timestamp_sec, 1)

    def test_runtime_interval_changes_are_per_camera(self):
        clock = FakeClock()
        with FrameSampler(self.video, "CAM_01", 2, clock=clock) as first, \
                FrameSampler(self.video, "CAM_02", 2, clock=clock) as second:
            first.read_due()
            second.read_due()
            first.set_interval(1)
            clock.now += 1
            self.assertAlmostEqual(first.read_due().video_timestamp_sec, 1)
            self.assertIsNone(second.read_due())
            self.assertEqual(second.interval_sec, 2)
            first.set_interval(2)
            clock.now += 1
            self.assertIsNone(first.read_due())
            self.assertEqual(second.read_due().camera_id, "CAM_02")
            clock.now += 1
            self.assertAlmostEqual(first.read_due().video_timestamp_sec, 3)

    def test_slow_consumer_skips_stale_samples(self):
        clock = FakeClock()
        with FrameSampler(self.video, clock=clock) as sampler:
            sampler.read_due()
            clock.now += 3.4
            sample = sampler.read_due()
            self.assertAlmostEqual(sample.video_timestamp_sec, 3.4)
            self.assertEqual(sample.frame_id, "CAM_01-000002")
            self.assertIsNone(sampler.read_due())

    def test_missing_backend_timestamps_fall_back_to_frame_index(self):
        original_get = cv2.VideoCapture.get

        def without_timestamps(capture, prop):
            if prop == cv2.CAP_PROP_POS_MSEC:
                return 0.0
            return original_get(capture, prop)

        with patch.object(cv2.VideoCapture, "get", without_timestamps):
            with FrameSampler(self.video) as sampler:
                samples = list(sampler.samples(realtime=False))
        self.assertEqual(len(samples), 6)
        for expected, sample in enumerate(samples):
            self.assertAlmostEqual(sample.video_timestamp_sec, expected)

    def test_fractional_intervals_do_not_accumulate_rounding_drift(self):
        with FrameSampler(self.video, interval_sec=0.35) as sampler:
            samples = list(sampler.samples(realtime=False))
        for index, sample in enumerate(samples):
            self.assertGreaterEqual(sample.video_timestamp_sec + 1e-9, index * 0.35)
            self.assertLess(sample.video_timestamp_sec - index * 0.35, 0.1 + 1e-9)

    def test_invalid_configuration_and_sources_fail_explicitly(self):
        for interval in (0, -1, float("nan"), float("inf")):
            with self.subTest(interval=interval):
                with self.assertRaises(ValueError):
                    FrameSampler(self.video, interval_sec=interval)
        with self.assertRaises(ValueError):
            FrameSampler(self.video, "CAM_99")
        with self.assertRaises(FileNotFoundError):
            FrameSampler(Path(self.temp.name) / "missing.avi")
        bad = Path(self.temp.name) / "not-video.txt"
        bad.write_text("not a video")
        with self.assertRaises(ValueError):
            FrameSampler(bad)
        with patch.object(cv2.VideoCapture, "get", return_value=0):
            with self.assertRaises(ValueError):
                FrameSampler(self.video)
        with FrameSampler(self.video) as sampler:
            with self.assertRaises(ValueError):
                sampler.set_interval(0)
            self.assertEqual(sampler.interval_sec, 1)

    def test_resources_are_closed_after_exceptions_and_eof(self):
        sampler = FrameSampler(self.video)
        with self.assertRaisesRegex(RuntimeError, "consumer failed"):
            with sampler:
                raise RuntimeError("consumer failed")
        self.assertFalse(sampler._capture.isOpened())
        self.assertIsNone(sampler.read_due())
        sampler.close()
        with FrameSampler(self.video) as sampler:
            list(sampler.samples(realtime=False))
            self.assertFalse(sampler._capture.isOpened())
            self.assertIsNone(sampler.read_due())

    def test_cli_extracts_two_cameras_and_jpegs(self):
        output = Path(self.temp.name) / "frames"
        result = subprocess.run(
            [sys.executable, "-m", "rsu.frame_automation", "--video", f"CAM_01={self.video}",
             "--video", f"CAM_02={self.video}", "--interval-sec", "2", "--no-realtime",
             "--output-dir", str(output)], capture_output=True, text=True, check=True,
        )
        rows = [json.loads(row) for row in result.stdout.splitlines()]
        self.assertEqual(len(rows), 6)
        self.assertEqual({r["camera_id"] for r in rows}, {"CAM_01", "CAM_02"})
        self.assertEqual(len(list(output.glob("*.jpg"))), 6)
        for row in rows:
            self.assertIsNotNone(cv2.imread(str(output / f"{row['frame_id']}.jpg")))

    def test_wall_clock_iterator_keeps_one_second_cadence(self):
        times = []
        with FrameSampler(self.video) as sampler:
            started = time.monotonic()
            for sample in sampler.samples():
                times.append(time.monotonic() - started)
                if len(times) == 3:
                    break
        self.assertEqual(len(times), 3)
        for expected, actual in enumerate(times):
            self.assertGreaterEqual(actual, expected - 0.02)
            self.assertLess(actual, expected + 0.5)


if __name__ == "__main__":
    unittest.main()
