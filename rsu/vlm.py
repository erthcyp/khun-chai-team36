"""OpenAI CLIP ViT-L/14 with the baseline's K=30 class prototypes."""

from pathlib import Path
from time import perf_counter

import numpy as np
import open_clip
import torch
from PIL import Image

from .decision_engine import CLASS_ORDER, normalize_scores

MODEL_NAME = "ViT-L-14"
DEFAULT_PROMPTS = Path(__file__).resolve().parents[1] / "assets" / "prompts" / "ape_k30.md"
CATEGORY_HEADINGS = {label.replace("_", " ").upper(): label for label in CLASS_ORDER}
CATEGORY_HEADINGS["SMOOTH FLOW"] = "normal_flow"


def parse_prompt_file(path: Path) -> dict[str, list[str]]:
    prompts = {label: [] for label in CLASS_ORDER}
    label = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("[") and "]" in line:
            label = CATEGORY_HEADINGS.get(line[1 : line.index("]")].strip().upper())
        elif line and label is not None:
            prompts[label].append(line.removeprefix("- ").strip())

    missing = [label for label, entries in prompts.items() if not entries]
    if missing:
        raise ValueError(f"Missing prompts in {path}: {', '.join(missing)}")
    return prompts


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class ClipPredictor:
    def __init__(self, prompt_file: Path = DEFAULT_PROMPTS, device: str | None = None) -> None:
        prompts = parse_prompt_file(prompt_file)
        self.device = torch.device(device or choose_device())
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained="openai", device=self.device
        )
        self.model.eval()
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)

        with torch.inference_mode():
            prototypes = []
            for label in CLASS_ORDER:
                tokens = tokenizer(prompts[label]).to(self.device)
                features = normalize_features(self.model.encode_text(tokens).float())
                prototype = normalize_features(features.mean(dim=0, keepdim=True))
                prototypes.append(prototype.squeeze(0))
            self.prototypes = torch.stack(prototypes)

    def predict(self, frame: Image.Image | np.ndarray) -> dict:
        """Return scores from an RGB PIL image or an OpenCV BGR uint8 frame."""
        started = perf_counter()
        if isinstance(frame, Image.Image):
            image = frame.convert("RGB")
        else:
            if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
                raise ValueError("Expected an H×W×3 OpenCV BGR uint8 frame.")
            image = Image.fromarray(frame[..., ::-1])

        with torch.inference_mode():
            pixels = self.preprocess(image).unsqueeze(0).to(self.device)
            feature = normalize_features(self.model.encode_image(pixels).float())
            cosines = (feature @ self.prototypes.T).squeeze(0).cpu().tolist()

        scores = normalize_scores(cosines)
        best = max(range(len(scores)), key=scores.__getitem__)
        return {
            "prediction": CLASS_ORDER[best],
            "scores": scores,
            "score_order": list(CLASS_ORDER),
            "confidence": scores[best],
            "inference_ms": (perf_counter() - started) * 1000,
            "model_name": MODEL_NAME,
        }
