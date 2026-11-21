"""RSU sender: infer traffic state at the edge, send the 16-byte vector to the TCC.

Usage:
  python -m apps.rsu.main --mock                       # scripted demo, no ML deps
  python -m apps.rsu.main --images data/samples/       # real CLIP + APE + CPV
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import httpx

from semcom.payload import encode, top_class


def iter_images(folder: Path):
    from PIL import Image

    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    while True:
        for p in files:
            yield p.name, Image.open(p).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcc", default=os.getenv("TCC_URL", "http://localhost:8000"))
    ap.add_argument("--rsu-id", default=os.getenv("RSU_ID", "RSU-01"))
    ap.add_argument("--interval", type=float, default=float(os.getenv("RSU_INTERVAL", "3")))
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--images", type=Path)
    args = ap.parse_args()

    if args.mock or not args.images:
        from apps.rsu.engine import MockEngine

        engine, frames = MockEngine(), iter(lambda: ("mock", None), None)
    else:
        from apps.rsu.engine import ClipEngine

        engine, frames = ClipEngine(), iter_images(args.images)

    with httpx.Client(timeout=5) as client:
        for seq, (name, img) in enumerate(frames):
            m, info = engine.infer(img)
            body = encode(m)
            t_send = time.perf_counter()
            try:
                r = client.post(
                    f"{args.tcc}/ingest",
                    content=body,
                    headers={
                        "Content-Type": "application/octet-stream",
                        "X-RSU-ID": args.rsu_id,
                        "X-Seq": str(seq),
                        "X-Send-Ms": str(int(time.time() * 1000)),
                        "X-CPV": "swapped" if info["swapped"] else ("checked" if info["triggered"] else "skip"),
                        "X-Frame": name,
                    },
                )
                rtt_ms = (time.perf_counter() - t_send) * 1000
                cls, conf = top_class(m)
                print(f"[{args.rsu_id}#{seq}] {cls:<13} {conf:.2f} cpv={info['triggered']}/{info['swapped']} "
                      f"one-way≈{rtt_ms/2:.2f}ms http={r.status_code}")
            except httpx.HTTPError as e:
                print(f"[{args.rsu_id}#{seq}] send failed: {e}")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
