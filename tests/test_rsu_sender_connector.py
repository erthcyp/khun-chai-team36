"""Exercise the runner-to-sender path with decoded frames and in-process HTTP."""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import httpx
import numpy as np

from rsu.decision_engine import CLASS_ORDER
from rsu.main import make_parser, run_predictions, run_streams
from rsu.sender_connector import SenderConnector, load_locations


class TestPredictor:
    def predict(self, frame):
        # The fixture colors make the two test cameras distinguishable.
        scores = [0.8, 0.1, 0.06, 0.04] if frame.mean() < 128 else [0.1, 0.8, 0.06, 0.04]
        return {
            "prediction": CLASS_ORDER[scores.index(max(scores))],
            "scores": scores,
            "score_order": list(CLASS_ORDER),
            "confidence": max(scores),
            "inference_ms": 1.0,
        }


class SenderConnectorTests(unittest.TestCase):
    def test_video_runners_send_contract_payload_and_jpeg(self):
        for streams in (1, 2, 6):
            with self.subTest(streams=streams), tempfile.TemporaryDirectory() as directory:
                self.check_delivery(Path(directory), streams)

    def check_delivery(self, root, streams):
        videos = []
        cameras = []
        for index in range(streams):
            camera_id = f"CAM_{index + 1:02d}"
            cameras.append({"camera_id": camera_id, "location": {"label": f"Junction {index}"}})
            path = root / f"clip-{index}.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
            self.assertTrue(writer.isOpened())
            try:
                for _ in range(8):
                    writer.write(np.full((48, 64, 3), 40 if index == 0 else 210, dtype=np.uint8))
            finally:
                writer.release()
            videos.extend(["--video", str(path)])

        config = root / "cameras.json"
        config.write_text(json.dumps(cameras))
        locations = load_locations(config, tuple(camera["camera_id"] for camera in cameras))
        semantic, snapshots = [], []

        async def receive(request):
            if request.url.path == "/api/v1/semantic":
                semantic.append(json.loads(request.content))
            else:
                self.assertEqual(request.url.path, "/api/v1/snapshot")
                body = await request.aread()
                meta = body.split(b'name="meta"\r\n\r\n', 1)[1].split(b"\r\n", 1)[0]
                snapshots.append(json.loads(meta))
                self.assertIn(b"\xff\xd8", body)
            return httpx.Response(200)

        args = make_parser().parse_args(videos + ["--max-frames", "1"])
        runner = run_predictions if streams == 1 else run_streams
        with SenderConnector("http://test", locations, transport=httpx.MockTransport(receive)) as connector:
            runner(args, TestPredictor(), emit=lambda _: None, connector=connector)

        self.assertEqual(len(semantic), streams)
        self.assertEqual(len(snapshots), streams)
        by_frame = {item["frame_id"]: item for item in snapshots}
        for item in semantic:
            camera_id = item["camera_id"]
            self.assertEqual(item["schema_version"], "1.0")
            self.assertEqual(item["message_type"], "semantic_update")
            self.assertEqual(item["score_order"], list(CLASS_ORDER))
            self.assertEqual(item["location"], locations[camera_id])
            self.assertEqual(item["confidence"], max(item["scores"]))
            self.assertEqual(datetime.fromisoformat(item["captured_at"]).tzinfo, timezone.utc)
            self.assertEqual(item["sampling_interval_sec"], 2.0 if camera_id == "CAM_01" else 1.0)
            matched = by_frame[item["frame_id"]]
            self.assertEqual(matched["camera_id"], camera_id)
            self.assertEqual(matched["event_id"], item["event_id"])
            self.assertEqual(matched["run_id"], item["run_id"])
            self.assertEqual(matched["captured_at"], item["captured_at"])
