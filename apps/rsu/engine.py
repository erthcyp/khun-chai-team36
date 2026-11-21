"""RSU inference engines.

- MockEngine: no ML deps; produces realistic 4-D vectors so the whole pipeline
  (RSU -> TCC -> agent -> dashboard) can be demoed and developed in parallel.
- ClipEngine: real APE inference (paper Eq. 4-6) with open_clip, plus CPV partners.
  Needs `pip install -r requirements-clip.txt` and prompts/ape_prompts.json.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from semcom.cpv import verify
from semcom.payload import CLASSES

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "prompts" / "ape_prompts.json"


class MockEngine:
    """Cycles through scripted scenarios so the demo is predictable."""

    SCENARIOS = {
        "normal_flow": [0.86, 0.08, 0.03, 0.03],
        "congestion": [0.12, 0.78, 0.06, 0.04],
        "accident": [0.10, 0.14, 0.71, 0.05],
        "invalid_scene": [0.05, 0.04, 0.03, 0.88],
        "ambiguous": [0.44, 0.08, 0.40, 0.08],  # triggers CPV
    }

    def __init__(self, script: list[str] | None = None, seed: int = 0):
        self.script = script or ["normal_flow"] * 4 + ["congestion"] * 2 + ["ambiguous", "accident"]
        self.rng = random.Random(seed)
        self.i = 0

    def infer(self, image=None) -> tuple[np.ndarray, dict]:
        name = self.script[self.i % len(self.script)]
        self.i += 1
        base = np.array(self.SCENARIOS[name], dtype=np.float32)
        noise = np.array([self.rng.uniform(-0.02, 0.02) for _ in range(4)], dtype=np.float32)
        m = np.clip(base + noise, 1e-4, None)
        m = m / m.sum()
        # mock partners: in the ambiguous case both vote for "accident"
        acc = CLASSES.index("accident")

        def partner(_img, pair):
            return acc if acc in pair else pair[0]

        return verify(m, image, [partner, partner])


class ClipEngine:
    """Primary + partner VLMs, each with APE class embeddings (Eq. 5)."""

    # (open_clip model name, pretrained tag). Paper: CLIP ViT-L-14 primary,
    # partners from the other families at the same scale.
    PRIMARY = ("ViT-L-14", "openai")
    PARTNERS = [("ViT-L-14", "laion2b_s32b_b82k"), ("ViT-L-16-SigLIP2-256", "webli")]

    def __init__(self, primary=None, partners=None, device: str | None = None):
        import open_clip  # noqa: F401  (lazy heavy import)
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if torch.backends.mps.is_available() else "cpu")
        prompts = json.loads(PROMPTS.read_text())
        self.prompts = [prompts[c] for c in CLASSES]
        self.primary = self._load(*(primary or self.PRIMARY))
        self.partners = [self._load(*p) for p in (partners or self.PARTNERS)]

    def _load(self, name: str, tag: str):
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=tag, device=self.device)
        tokenizer = open_clip.get_tokenizer(name)
        model.eval()
        with self.torch.no_grad():
            class_emb = []
            for plist in self.prompts:
                t = model.encode_text(tokenizer(plist).to(self.device))
                t = t / t.norm(dim=-1, keepdim=True)
                class_emb.append(t.mean(0))  # Eq. 5: average of K prompt embeddings
            T = self.torch.stack(class_emb)
            T = T / T.norm(dim=-1, keepdim=True)
        return {"model": model, "preprocess": preprocess, "T": T}

    def _vector(self, vlm, image) -> np.ndarray:
        with self.torch.no_grad():
            x = vlm["preprocess"](image).unsqueeze(0).to(self.device)
            v = vlm["model"].encode_image(x)
            v = v / v.norm(dim=-1, keepdim=True)
            s = (v @ vlm["T"].T).squeeze(0)       # cosine similarity
            m = (100 * s).softmax(dim=-1)          # Eq. 6, scale 100
        return m.float().cpu().numpy()

    def infer(self, image) -> tuple[np.ndarray, dict]:
        m = self._vector(self.primary, image)

        def make_partner(vlm):
            def vote(img, pair):
                pm = self._vector(vlm, img)
                return pair[0] if pm[pair[0]] >= pm[pair[1]] else pair[1]
            return vote

        return verify(m, image, [make_partner(p) for p in self.partners])
