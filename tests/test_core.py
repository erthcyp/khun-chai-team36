import numpy as np
import pytest
from fastapi.testclient import TestClient

from apps.rsu.engine import MockEngine
from apps.tcc.main import app
from semcom.cpv import needs_verification, verify
from semcom.payload import PAYLOAD_SIZE, decode, encode


def test_payload_is_16_bytes_and_roundtrips():
    m = [0.7, 0.1, 0.15, 0.05]
    b = encode(m)
    assert len(b) == PAYLOAD_SIZE == 16
    assert np.allclose(decode(b), m, atol=1e-6)


def test_decode_rejects_bad_payload():
    with pytest.raises(ValueError):
        decode(b"\x00" * 15)


def test_cpv_skips_confident_prediction():
    m, info = verify([0.8, 0.1, 0.05, 0.05], None, [lambda *_: 1, lambda *_: 1])
    assert not info["triggered"] and m[0] == pytest.approx(0.8)


def test_cpv_swaps_only_when_both_partners_pick_top2():
    m = [0.44, 0.08, 0.40, 0.08]
    assert needs_verification(m)
    out, info = verify(m, None, [lambda *_: 2, lambda *_: 2])
    assert info["swapped"] and np.argmax(out) == 2 and out.shape == (4,)
    out, info = verify(m, None, [lambda *_: 2, lambda *_: 0])
    assert not info["swapped"] and np.argmax(out) == 0


def test_tcc_ingest_end_to_end():
    client = TestClient(app)
    m, _ = MockEngine(script=["accident"]).infer()
    r = client.post("/ingest", content=encode(m), headers={"X-RSU-ID": "RSU-T", "X-Seq": "1"})
    assert r.status_code == 204
    assert client.get("/api/rsus").json()["RSU-T"]["state"] == "accident"
    assert client.post("/ingest", content=b"bad").status_code == 400
