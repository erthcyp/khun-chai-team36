"""Test real video decoding and throughput accounting without loading model weights.

The predictor doubles below exist only in unit tests; the CLI always uses CLIP.
"""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from rsu.main import (LocalStream, Throughput, grab_stream_frame, letterbox_frame,
                      main, make_parser, preview_image, run_predictions, run_preview,
                      run_streams, source_paths)
from rsu.vision_review import Verdict


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class PredictorDouble:
    def __init__(self, clock, duration=0.2, labels=None):
        self.clock = clock
        self.duration = duration
        self.labels = iter(labels) if labels else None
        self.frames_seen = 0

    def predict(self, frame):
        if frame.shape != (48, 64, 3) or frame.dtype != np.uint8:
            raise AssertionError("Expected a decoded BGR uint8 frame")
        self.frames_seen += 1
        self.clock.now += self.duration
        return {"prediction": next(self.labels) if self.labels else "normal_flow",
                "inference_ms": self.duration * 1000}


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.video = Path(cls.temp.name) / "fixture.avi"
        writer = cv2.VideoWriter(str(cls.video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
        if not writer.isOpened():
            raise RuntimeError("MJPG encoder required")
        try:
            for index in range(41):
                writer.write(np.full((48, 64, 3), index * 5, dtype=np.uint8))
        finally:
            writer.release()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_video(self, *extra, duration=0.2, labels=None):
        args = make_parser().parse_args(["--video", str(self.video), *extra])
        clock = Clock()
        predictor = PredictorDouble(clock, duration=duration, labels=labels)
        records = []
        summary = run_predictions(args, predictor, emit=records.append, clock=clock, sleep=clock.sleep)
        return predictor, records, summary

    def test_realtime_reports_one_output_per_second_without_startup_inflation(self):
        predictor, records, summary = self.run_video("--max-frames", "3")
        self.assertEqual(predictor.frames_seen, 3)
        self.assertEqual([round(r["video_timestamp_sec"]) for r in records[:-1]], [0, 1, 2])
        self.assertAlmostEqual(summary["observed_output_fps"], 1)
        self.assertAlmostEqual(summary["model_call_rate_fps"], 5)
        self.assertAlmostEqual(summary["mean_prediction_ms"], 200)
        self.assertEqual(summary["frame_processing_overruns"], 0)
        self.assertIsNone(records[0]["output_gap_sec"])
        self.assertAlmostEqual(records[1]["output_gap_sec"], 1)
        self.assertEqual(records[-1]["type"], "summary")
        self.assertEqual(summary["status"], "completed")

    def test_offline_mode_measures_capacity_without_playback_waits(self):
        _, records, summary = self.run_video("--no-realtime", "--max-frames", "3")
        self.assertEqual(summary["mode"], "offline")
        self.assertAlmostEqual(summary["observed_output_fps"], 5)
        self.assertAlmostEqual(records[1]["video_timestamp_sec"], 1)

    def test_slow_inference_reports_overruns_and_skips_stale_video(self):
        _, records, summary = self.run_video("--max-frames", "3", duration=1.6)
        self.assertEqual(summary["frame_processing_overruns"], 3)
        self.assertAlmostEqual(summary["observed_output_fps"], 0.625)
        self.assertGreater(records[2]["video_timestamp_sec"], 3)

    def test_filter_saves_matching_frames_but_counts_every_prediction(self):
        output = Path(self.temp.name) / "invalid"
        predictor, records, summary = self.run_video(
            "--no-realtime", "--max-frames", "3", "--only-class", "invalid_scene",
            "--output-dir", str(output), labels=["normal_flow", "invalid_scene", "congestion"],
        )
        self.assertEqual(predictor.frames_seen, 3)
        self.assertEqual(summary["predictions"], 3)
        self.assertEqual(summary["printed_predictions"], 1)
        self.assertEqual(summary["saved_images"], 1)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["frame_id"], "CAM_01-000002")
        image = cv2.imread(records[0]["image_path"])
        self.assertIsNotNone(image)
        self.assertAlmostEqual(float(image.mean()), 50, delta=2)

    def test_eof_completes_without_a_frame_limit(self):
        _, _, summary = self.run_video("--no-realtime", "--interval-sec", "2")
        self.assertEqual(summary["predictions"], 3)
        self.assertEqual(summary["target_output_fps"], 0.5)
        self.assertAlmostEqual(summary["last_video_timestamp_sec"], 4)

    def test_vision_review_uses_following_frame_and_resolves_only_the_target(self):
        clock = Clock()

        class CloseScorePredictor(PredictorDouble):
            def predict(self, frame):
                record = super().predict(frame)
                record["scores"] = (
                    [0.49, 0.02, 0.48, 0.01] if self.frames_seen == 2 else [0.9, 0.03, 0.04, 0.03]
                )
                return record

        args = make_parser().parse_args(["--video", str(self.video), "--openai-review", "--no-realtime"])
        records = []
        run_predictions(
            args, CloseScorePredictor(clock), emit=records.append, clock=clock,
            reviewer=lambda _: Verdict("accident", "Visible contact at the target time."),
        )
        predictions = [record for record in records if record["type"] == "prediction"]
        review, = [record for record in records if record["type"] == "verification"]
        self.assertEqual(predictions[1]["review_status"], "pending")
        self.assertEqual(predictions[1]["next_sampling_interval_sec"], 1)
        self.assertEqual(predictions[2]["prediction"], "normal_flow")
        self.assertEqual(review["frame_id"], predictions[1]["frame_id"])
        self.assertEqual(review["reviewed_class"], "accident")
        self.assertEqual([frame["video_timestamp_sec"] for frame in review["evidence_frames"]], [0, 2, 3])
        self.assertTrue(all(record["alert"] is None for record in predictions))

    def test_errors_emit_a_failed_summary_and_propagate(self):
        args = make_parser().parse_args(["--video", str(self.video)])
        records = []

        class FailedPredictor:
            def predict(self, frame):
                raise RuntimeError("inference failed")

        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            run_predictions(args, FailedPredictor(), emit=records.append)
        self.assertEqual(records[-1]["status"], "failed")
        self.assertEqual(records[-1]["predictions"], 0)

    def test_interrupt_emits_summary(self):
        args = make_parser().parse_args(["--video", str(self.video)])
        records = []

        class InterruptedPredictor:
            def predict(self, frame):
                raise KeyboardInterrupt

        summary = run_predictions(args, InterruptedPredictor(), emit=records.append)
        self.assertEqual(summary["status"], "interrupted")

    def test_missing_video_fails_before_model_loading(self):
        result = subprocess.run(
            [sys.executable, "-m", "rsu.main", "--video", str(self.video.parent / "missing.mp4")],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("Loading VLM", result.stderr)

    def test_help_lists_video_option(self):
        result = subprocess.run([sys.executable, "-m", "rsu.main", "--help"],
                                capture_output=True, text=True, check=True)
        self.assertIn("--video", result.stdout)
        self.assertIn("--max-frames", result.stdout)

    def test_bad_limits_fail_argument_validation(self):
        for flag, value in (("--interval-sec", "0"), ("--interval-sec", "nan"),
                            ("--interval-sec", "inf"), ("--max-frames", "0")):
            with self.subTest(flag=flag, value=value), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(["--video", str(self.video), flag, value])

    def test_stop_event_ends_wait_without_another_prediction(self):
        args = make_parser().parse_args(["--video", str(self.video)])
        stop = threading.Event()
        records = []
        clock = Clock()
        predictor = PredictorDouble(clock)
        summary = run_predictions(args, predictor, emit=records.append,
                                  on_prediction=lambda record: stop.set(), stop_event=stop)
        self.assertEqual(summary["status"], "stopped")
        self.assertEqual(summary["predictions"], 1)

    def test_preview_observer_receives_predictions_excluded_by_filter(self):
        args = make_parser().parse_args(["--video", str(self.video), "--no-realtime",
                                         "--only-class", "invalid_scene", "--max-frames", "2"])
        clock = Clock()
        predictor = PredictorDouble(clock)
        observed = []
        summary = run_predictions(args, predictor, emit=lambda record: None,
                                  clock=clock, on_prediction=observed.append)
        self.assertEqual(len(observed), 2)
        self.assertEqual(summary["printed_predictions"], 0)

    def test_preview_requires_realtime(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--video", str(self.video), "--preview", "--no-realtime"])

    def test_gui_keeps_polling_while_predictor_waits_in_another_thread(self):
        args = make_parser().parse_args(["--video", str(self.video), "--preview", "--max-frames", "1"])
        entered = threading.Event()
        release = threading.Event()
        inference_threads = []
        gui_threads = []
        polls_during_inference = []
        records = []

        class WaitingPredictor:
            def predict(self, frame):
                inference_threads.append(threading.current_thread())
                entered.set()
                if not release.wait(2):
                    raise RuntimeError("GUI stopped polling during inference")
                return {"prediction": "normal_flow", "confidence": 0.7}

        def poll(delay):
            gui_threads.append(threading.current_thread())
            if entered.is_set():
                polls_during_inference.append(1)
                if len(polls_during_inference) >= 3:
                    release.set()
            time.sleep(0.005)
            return -1

        with patch.object(cv2, "namedWindow"), patch.object(cv2, "resizeWindow"), \
                patch.object(cv2, "imshow"), patch.object(cv2, "destroyWindow") as destroy, \
                patch.object(cv2, "getWindowProperty", return_value=1), \
                patch.object(cv2, "waitKey", side_effect=poll):
            summary = run_preview(args, WaitingPredictor(), emit=records.append)
        self.assertGreaterEqual(len(polls_during_inference), 3)
        self.assertTrue(all(t is threading.main_thread() for t in gui_threads))
        self.assertTrue(all(t is not threading.main_thread() for t in inference_threads))
        self.assertEqual(summary["predictions"], 1)
        self.assertTrue(summary["preview"])
        destroy.assert_called_once()

    def test_preview_quit_stops_worker_and_closes_window(self):
        args = make_parser().parse_args(["--video", str(self.video), "--preview"])
        entered = threading.Event()

        class BriefPredictor:
            def predict(self, frame):
                entered.set()
                time.sleep(0.02)
                return {"prediction": "normal_flow"}

        def quit_key(delay):
            if not entered.wait(2):
                raise RuntimeError("Prediction worker did not start")
            return ord("q")

        with patch.object(cv2, "namedWindow"), patch.object(cv2, "resizeWindow"), \
                patch.object(cv2, "imshow"), patch.object(cv2, "destroyWindow") as destroy, \
                patch.object(cv2, "waitKey", side_effect=quit_key):
            summary = run_preview(args, BriefPredictor(), emit=lambda record: None)
        self.assertEqual(summary["status"], "stopped")
        self.assertEqual(summary["predictions"], 1)
        destroy.assert_called_once()
        self.assertFalse(any(t.name == "roadsense-inference" for t in threading.enumerate()))

    def test_preview_reports_worker_failure(self):
        args = make_parser().parse_args(["--video", str(self.video), "--preview"])

        class FailedPredictor:
            def predict(self, frame):
                raise RuntimeError("worker inference failed")

        with patch.object(cv2, "namedWindow"), patch.object(cv2, "resizeWindow"), \
                patch.object(cv2, "imshow"), patch.object(cv2, "destroyWindow"), \
                patch.object(cv2, "getWindowProperty", return_value=1), \
                patch.object(cv2, "waitKey", return_value=-1):
            with self.assertRaisesRegex(RuntimeError, "worker inference failed"):
                run_preview(args, FailedPredictor(), emit=lambda record: None)

    def test_overlay_resizes_without_modifying_input_pixels(self):
        original = np.full((1080, 1920, 3), 50, dtype=np.uint8)
        image = preview_image(original, 2, "CAM_01", {
            "prediction": "normal_flow", "confidence": 0.8,
            "video_timestamp_sec": 1, "prediction_wall_ms": 100,
        })
        self.assertEqual(image.shape, (634, 960, 3))
        self.assertTrue(np.all(image[94:] == 50))  # Text is outside the complete video image.
        self.assertTrue(np.all(original == 50))

    def test_letterbox_preserves_portrait_and_ultrawide_frames(self):
        for height, width in ((200, 100), (100, 400)):
            with self.subTest(height=height, width=width):
                frame = np.full((height, width, 3), 60, dtype=np.uint8)
                frame[:10] = (0, 0, 255)
                frame[-10:] = (255, 0, 0)
                result = letterbox_frame(frame, (320, 180))
                scale = min(320 / width, 180 / height)
                expected_width, expected_height = round(width * scale), round(height * scale)
                left, top = (320 - expected_width) // 2, (180 - expected_height) // 2
                content = result[top:top + expected_height, left:left + expected_width]
                self.assertEqual(result.shape, (180, 320, 3))
                self.assertTrue(np.all(content[0] == (0, 0, 255)))
                self.assertTrue(np.all(content[-1] == (255, 0, 0)))
                # Non-black bounding box retains source aspect ratio within rounding.
                rows, cols = np.nonzero(np.any(result != 0, axis=2))
                self.assertEqual(cols.max() - cols.min() + 1, expected_width)
                self.assertEqual(rows.max() - rows.min() + 1, expected_height)

    def test_multistream_emits_each_result_and_uses_one_predictor(self):
        args = make_parser().parse_args(["--video", str(self.video), "--streams", "6",
                                         "--max-frames", "2", "--interval-sec", "0.05"])
        predictor = PredictorDouble(Clock())
        records = []
        summary = run_streams(args, predictor, emit=records.append)
        predictions = [r for r in records if r["type"] == "prediction"]
        self.assertEqual(predictor.frames_seen, 12)
        self.assertEqual(len(predictions), 12)
        self.assertEqual(summary["model_instances"], 1)
        self.assertEqual(summary["predictions"], 12)
        self.assertEqual(len({r["frame_id"] for r in predictions}), 12)
        self.assertEqual({r["stream_id"] for r in predictions},
                         {f"STREAM_{index:02d}" for index in range(1, 7)})
        self.assertTrue(all(s["predictions"] == 2 for s in summary["per_stream"]))
        self.assertTrue(all("queue_wait_ms" in r for r in predictions))
        self.assertTrue(all("camera_id" not in r for r in predictions))
        self.assertEqual(records[-1]["type"], "summary")

    def test_multistream_filters_output_but_counts_all_inferences(self):
        output = Path(self.temp.name) / "multi-filter"
        args = make_parser().parse_args(["--video", str(self.video), "--streams", "2", "--max-frames", "1",
                                         "--only-class", "invalid_scene", "--output-dir", str(output)])
        predictor = PredictorDouble(Clock(), labels=["normal_flow", "invalid_scene"])
        records = []
        summary = run_streams(args, predictor, emit=records.append)
        predictions = records[:-1]
        self.assertEqual(predictor.frames_seen, 2)
        self.assertEqual(summary["predictions"], 2)
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["prediction"], "invalid_scene")
        self.assertTrue(Path(predictions[0]["image_path"]).is_file())
        self.assertEqual(sum(s["saved_images"] for s in summary["per_stream"]), 1)

    def test_multistream_preview_remains_responsive_and_quits(self):
        args = make_parser().parse_args(["--video", str(self.video), "--streams", "2", "--preview", "--loop"])
        entered = threading.Event()
        release = threading.Event()
        polls = []
        records = []

        class WaitingPredictor:
            def predict(self, frame):
                entered.set()
                if not release.wait(2):
                    raise RuntimeError("GUI failed to poll during inference")
                return {"prediction": "normal_flow"}

        def quit_after_polling(delay):
            if entered.is_set():
                polls.append(threading.current_thread())
                if len(polls) >= 3:
                    release.set()
                    return ord("q")
            time.sleep(0.005)
            return -1

        with patch.object(cv2, "namedWindow"), patch.object(cv2, "resizeWindow"), \
                patch.object(cv2, "imshow") as show, patch.object(cv2, "destroyWindow"), \
                patch.object(cv2, "getWindowProperty", return_value=1), \
                patch.object(cv2, "waitKey", side_effect=quit_after_polling):
            summary = run_streams(args, WaitingPredictor(), emit=records.append)
        self.assertEqual(summary["status"], "stopped")
        self.assertTrue(all(t is threading.main_thread() for t in polls))
        self.assertGreater(show.call_count, 0)
        canvas = show.call_args.args[1]
        self.assertEqual(canvas.shape, (252, 640, 3))
        self.assertFalse(any(t.name == "roadsense-shared-vlm" for t in threading.enumerate()))

    def test_multistream_replaces_stale_pending_frames_when_predictor_is_slow(self):
        args = make_parser().parse_args(["--video", str(self.video), "--streams", "2",
                                         "--max-frames", "3", "--interval-sec", "0.01"])

        class SlowPredictor:
            def predict(self, frame):
                time.sleep(0.09)
                return {"prediction": "normal_flow"}

        summary = run_streams(args, SlowPredictor(), emit=lambda record: None)
        self.assertEqual(summary["predictions"], 6)
        self.assertGreater(sum(s["replaced_pending_samples"] for s in summary["per_stream"]), 0)

    def test_multistream_worker_failure_is_reported(self):
        args = make_parser().parse_args(["--video", str(self.video), "--streams", "2"])
        records = []

        class FailedPredictor:
            def predict(self, frame):
                raise RuntimeError("shared VLM failed")

        with self.assertRaisesRegex(RuntimeError, "shared VLM failed"):
            run_streams(args, FailedPredictor(), emit=records.append)
        self.assertEqual(records[-1]["status"], "failed")
        self.assertFalse(any(t.name == "roadsense-shared-vlm" for t in threading.enumerate()))

    def test_multistream_rejects_offline_before_loading_model(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--video", str(self.video), "--streams", "6", "--no-realtime"])

    def test_multistream_eof_drains_pending_predictions(self):
        short = Path(self.temp.name) / "one-frame.avi"
        writer = cv2.VideoWriter(str(short), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
        self.assertTrue(writer.isOpened())
        writer.write(np.full((48, 64, 3), 50, dtype=np.uint8))
        writer.release()
        args = make_parser().parse_args(["--video", str(short), "--streams", "2"])
        summary = run_streams(args, PredictorDouble(Clock()), emit=lambda record: None)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["predictions"], 2)

    def test_six_different_sources_keep_frames_and_filenames_separate(self):
        paths = []
        arguments = []
        expected_pixels = []
        for index in range(6):
            path = Path(self.temp.name) / f"different-{index}.avi"
            paths.append(path)
            arguments.extend(["--video", str(path)])
            size = (64, 96) if index % 2 else (96, 64)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5 + index, size)
            self.assertTrue(writer.isOpened())
            value = 30 + index * 25
            writer.write(np.full((size[1], size[0], 3), value, dtype=np.uint8))
            writer.release()
            expected_pixels.append(value)
        observed = []

        class CollectingPredictor:
            def predict(self, frame):
                observed.append((float(frame.mean()), frame.shape))
                return {"prediction": "normal_flow"}

        args = make_parser().parse_args([*arguments, "--max-frames", "1"])
        records = []
        summary = run_streams(args, CollectingPredictor(), emit=records.append)
        self.assertEqual(summary["streams"], 6)
        self.assertEqual(summary["predictions"], 6)
        self.assertEqual(summary["videos"], [str(path) for path in paths])
        self.assertEqual([r["video"] for r in records[:-1]], [str(path) for path in paths])
        for expected, (actual, shape) in zip(expected_pixels, observed):
            self.assertAlmostEqual(actual, expected, delta=2)
        self.assertEqual({shape for _, shape in observed}, {(64, 96, 3), (96, 64, 3)})

    def test_short_clip_can_end_while_longer_clip_reaches_its_frame_limit(self):
        paths = []
        for name, count in (("early-end", 1), ("longer", 300)):
            path = Path(self.temp.name) / f"{name}.avi"
            paths.append(path)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
            self.assertTrue(writer.isOpened())
            for _ in range(count):
                writer.write(np.full((48, 64, 3), 60, dtype=np.uint8))
            writer.release()
        args = make_parser().parse_args(["--video", str(paths[0]), "--video", str(paths[1]),
                                         "--max-frames", "3", "--interval-sec", "0.05"])
        summary = run_streams(args, PredictorDouble(Clock()), emit=lambda record: None)
        first, second = summary["per_stream"]
        self.assertEqual(first["predictions"], 1)
        self.assertTrue(first["source_exhausted"])
        self.assertEqual(second["predictions"], 3)
        self.assertFalse(second["source_exhausted"])

    def test_every_source_is_validated_before_loading_the_model(self):
        result = subprocess.run(
            [sys.executable, "-m", "rsu.main", "--video", str(self.video),
             "--video", str(self.video.parent / "missing-second.mp4")], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("missing-second.mp4", result.stderr)
        self.assertNotIn("Loading VLM", result.stderr)

    def test_repeated_video_arguments_and_copy_mode_are_compatible(self):
        parser = make_parser()
        different = parser.parse_args(["--video", "one.mp4", "--video", "two.mp4"])
        self.assertEqual(source_paths(different), (Path("one.mp4"), Path("two.mp4")))
        copies = parser.parse_args(["--video", "one.mp4", "--streams", "6"])
        self.assertEqual(source_paths(copies), (Path("one.mp4"),) * 6)
        single = parser.parse_args(["--video", "single.mp4"])
        self.assertEqual(source_paths(single), (Path("single.mp4"),))

    def test_conflicting_counts_and_too_many_sources_are_rejected(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--video", "one.mp4", "--video", "two.mp4", "--streams", "6"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            make_parser().parse_args([value for index in range(7)
                                      for value in ("--video", f"{index}.mp4")])

    def make_replay_clip(self, name, values):
        path = Path(self.temp.name) / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
        self.assertTrue(writer.isOpened())
        for value in values:
            writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
        writer.release()
        return path

    def test_replay_reads_actual_beginning_again_without_resetting_global_position(self):
        path = self.make_replay_clip("replay-sequence.avi", [30, 70, 110])
        stream = LocalStream("STREAM_01", cv2.VideoCapture(str(path)), 10, Throughput(1), path)
        try:
            for index in range(10):
                self.assertTrue(grab_stream_frame(stream, True))
                ok, frame = stream.capture.retrieve()
                self.assertTrue(ok)
                self.assertAlmostEqual(float(frame.mean()), [30, 70, 110][index % 3], delta=2)
                self.assertEqual(stream.frame_index, index)
                self.assertEqual(stream.clip_frame_index, index % 3)
                self.assertEqual(stream.loop_index, index // 3)
                self.assertFalse(stream.eof)
        finally:
            stream.capture.release()

    def test_continuous_streams_keep_unique_ids_and_playback_time_across_replays(self):
        short = self.make_replay_clip("replay-short.avi", [40])
        longer = self.make_replay_clip("replay-longer.avi", [30, 70, 110])
        args = make_parser().parse_args(["--video", str(short), "--video", str(longer),
                                         "--loop", "--interval-sec", "0.05", "--max-frames", "5"])
        records = []
        summary = run_streams(args, PredictorDouble(Clock()), emit=records.append)
        predictions = records[:-1]
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["predictions"], 10)
        self.assertTrue(summary["loop"])
        self.assertEqual(len({row["frame_id"] for row in predictions}), 10)
        for stream_id in ("STREAM_01", "STREAM_02"):
            rows = [row for row in predictions if row["stream_id"] == stream_id]
            self.assertEqual(len(rows), 5)
            self.assertGreater(rows[-1]["loop_index"], 0)
            times = [row["playback_elapsed_sec"] for row in rows]
            self.assertEqual(times, sorted(times))
            self.assertEqual(rows[0]["loop_index"], 0)
        self.assertTrue(all(row["video_timestamp_sec"] < 0.3 for row in predictions))
        self.assertTrue(all(not item["source_exhausted"] for item in summary["per_stream"]))

    def test_replay_reopens_capture_when_backend_cannot_seek(self):
        path = self.make_replay_clip("replay-no-seek.avi", [50])
        args = make_parser().parse_args(["--video", str(path), "--streams", "2", "--loop",
                                         "--interval-sec", "0.05", "--max-frames", "3"])
        with patch.object(cv2.VideoCapture, "set", return_value=False):
            summary = run_streams(args, PredictorDouble(Clock()), emit=lambda row: None)
        self.assertEqual(summary["predictions"], 6)
        self.assertTrue(all(item["replays"] >= 2 for item in summary["per_stream"]))

    def test_empty_replay_source_fails_instead_of_spinning(self):
        args = make_parser().parse_args(["--video", str(self.video), "--streams", "2", "--loop"])
        records = []
        with patch.object(cv2.VideoCapture, "grab", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "No decodable frames"):
                run_streams(args, PredictorDouble(Clock()), emit=records.append)
        self.assertEqual(records[-1]["status"], "failed")

    def test_loop_rejects_offline_and_review_before_model_loading(self):
        for extra in (("--no-realtime",), ("--openai-review",)):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--video", str(self.video), "--loop", *extra])

    def test_single_video_loop_cli_reuses_one_model_across_replays(self):
        path = self.make_replay_clip("single-loop-cli.avi", [50])
        predictor = PredictorDouble(Clock())
        predictor.device = "cpu"
        output = io.StringIO()
        with patch("rsu.vlm.ClipPredictor", return_value=predictor) as factory, \
                redirect_stdout(output), redirect_stderr(io.StringIO()):
            main(["--video", str(path), "--loop", "--interval-sec", "0.05", "--max-frames", "2"])
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        factory.assert_called_once()
        self.assertEqual(rows[-1]["predictions"], 2)
        self.assertTrue(rows[-1]["loop"])
        self.assertGreater(rows[-2]["loop_index"], 0)


class ThroughputTests(unittest.TestCase):
    def test_measured_rates_and_percentiles(self):
        stats = Throughput(1)
        self.assertIsNone(stats.summary()["observed_output_fps"])
        stats.record(800, 1000, 10)
        self.assertIsNone(stats.summary()["observed_output_fps"])
        stats.record(1200, 1250, 11.2)
        summary = stats.summary()
        self.assertAlmostEqual(summary["observed_output_fps"], 1 / 1.2)
        self.assertAlmostEqual(summary["mean_prediction_ms"], 1000)
        self.assertAlmostEqual(summary["p95_prediction_ms"], 1180)
        self.assertEqual(summary["frame_processing_overruns"], 1)
        # JSON output must remain valid even before any predictions exist.
        json.dumps(Throughput(1).summary(), allow_nan=False)

    def test_latency_history_is_bounded(self):
        stats = Throughput(1)
        for index in range(1100):
            stats.record(index + 1, index + 1, index)
        self.assertEqual(stats.count, 1100)
        self.assertEqual(stats.summary()["p95_window_frames"], 1000)
        self.assertEqual(stats.prediction_window[0], 101)


if __name__ == "__main__":
    unittest.main()
