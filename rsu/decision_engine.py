"""Per-camera verification with confidence gates adapted from the RSU baseline."""

from dataclasses import dataclass
from math import exp, isfinite
from typing import Sequence
from uuid import uuid4

CLASS_ORDER = ("normal_flow", "congestion", "accident", "invalid_scene")
INCIDENTS = {
    "congestion": ("high", "Congestion detected"),
    "accident": ("critical", "Accident detected"),
}


def normalize_scores(cosine_scores: Sequence[float]) -> list[float]:
    """Apply the baseline's scaled softmax on the contract's 0–1 scale."""
    if len(cosine_scores) != len(CLASS_ORDER) or not all(map(isfinite, cosine_scores)):
        raise ValueError("Expected four finite cosine scores in RoadSense class order.")
    peak = max(cosine_scores)
    weights = [exp(100.0 * (score - peak)) for score in cosine_scores]
    total = sum(weights)
    return [weight / total for weight in weights]


@dataclass(frozen=True)
class DecisionConfig:
    high_confidence: float = 0.60
    minimum_confidence: float = 0.30
    minimum_lead: float = 0.30
    confirm_frames: int = 2
    recovery_frames: int = 3
    normal_interval_sec: float = 2.0
    verify_interval_sec: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_confidence <= self.high_confidence <= 1:
            raise ValueError("Confidence thresholds must be ordered within [0, 1].")
        if not 0 <= self.minimum_lead <= 1:
            raise ValueError("The score lead must be within [0, 1].")
        if self.confirm_frames < 2 or self.recovery_frames < 2:
            raise ValueError("Verification and recovery require at least two frames.")
        if not 0 < self.verify_interval_sec < self.normal_interval_sec:
            raise ValueError("The verification interval must be shorter than the normal interval.")


@dataclass(frozen=True)
class Decision:
    predicted_class: str
    verification_state: str
    sampling_interval_sec: float
    priority: str
    event_id: str | None = None
    alert: dict[str, str] | None = None


@dataclass
class _CameraState:
    verifying: bool = False
    normal_count: int = 0
    candidate: str | None = None
    candidate_count: int = 0
    event_class: str | None = None
    event_id: str | None = None
    confirmed: bool = False


class DecisionEngine:
    def __init__(self, config: DecisionConfig = DecisionConfig()) -> None:
        self.config = config
        self._cameras: dict[str, _CameraState] = {}

    def update(self, camera_id: str, scores: Sequence[float]) -> Decision:
        """Consume one fresh prediction in playback order for the given camera."""
        if len(scores) != len(CLASS_ORDER) or not all(0 <= score <= 1 for score in scores):
            raise ValueError("Expected four normalized scores in RoadSense class order.")

        state = self._cameras.setdefault(camera_id, _CameraState())
        top, runner_up, *_ = sorted(range(len(scores)), key=lambda index: -scores[index])
        label = CLASS_ORDER[top]
        confidence = scores[top]
        lead = confidence - scores[runner_up]
        config = self.config
        strong = confidence >= config.high_confidence
        reliable = strong or (
            confidence >= config.minimum_confidence and lead >= config.minimum_lead
        )

        if reliable and label != state.candidate:
            state.candidate = label
            state.candidate_count = 0
        if reliable:
            state.candidate_count += 1
        else:
            state.candidate = None
            state.candidate_count = 0

        if reliable and label == "normal_flow":
            state.normal_count += 1
            if not state.verifying or state.normal_count >= config.recovery_frames:
                self._cameras[camera_id] = _CameraState()
                return Decision(label, "normal", config.normal_interval_sec, "low")
            return self._decision(label, "recovery", state)

        state.normal_count = 0
        state.verifying = True
        if label == "invalid_scene":
            return Decision(label, "invalid", config.verify_interval_sec, "medium")

        confirmed = label in INCIDENTS and state.candidate_count >= config.confirm_frames
        preliminary = label == "accident" and strong and state.event_class != label
        if confirmed or preliminary:
            if state.event_class != label:
                state.event_class = label
                state.event_id = f"{camera_id}-evt-{uuid4().hex}"
            state.confirmed = confirmed

        phase = "alert" if state.confirmed and state.event_class == label else "verifying"
        return self._decision(label, phase, state)

    def _decision(self, label: str, phase: str, state: _CameraState) -> Decision:
        interval = self.config.verify_interval_sec
        if state.event_class is None:
            return Decision(label, phase, interval, "medium")
        priority, description = INCIDENTS[state.event_class]
        alert = {
            "level": "confirmed" if state.confirmed else "preliminary",
            "type": state.event_class,
            "message": description if state.confirmed else "Possible accident",
        }
        return Decision(label, phase, interval, priority, state.event_id, alert)
