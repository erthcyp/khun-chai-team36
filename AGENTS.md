# AGENTS.md — RoadSense (team khun-chai, TEAM36)

Codex reads this file automatically. Full context: `docs/master-plan.md` (read it before architecture or demo decisions).
This repo started EMPTY of application code at hacking start (10:00, 21 Nov 2026). Do not assume a module exists — inspect first.

## What we are building
RoadSense: edge traffic monitoring. **Mac 1 = simulated RSU** reads 2–3 traffic videos, samples frames per camera
(every 2 s normally, 1 s when verifying), classifies them with an existing VLM + APE, keeps a temporal decision state
per camera, and sends **semantic JSON** (plus optional async JPEG previews) to **Mac 2 = simulated TCC**, which shows
a live dashboard with alerts, camera priority, sampling interval, and measured AI-call reduction vs fixed 1 s sampling.

## Fixed conventions (never change without the whole team)
- Class order everywhere: `["normal_flow", "congestion", "accident", "invalid_scene"]`
- Payload / endpoints: `docs/api-contract.md` (single source of truth; `shared/schemas.py` implements it)
- Camera ids `CAM_01`, `CAM_02`, `CAM_03`. Timestamps: `video_timestamp_sec` (playback) + ISO wall-clock fields.
- The adaptive feedback loop (decision → sampler) runs on Mac 1 only.

## Layout (create as you go; owner in brackets)
- `rsu/` [Mac 1] `frame_automation.py` (P1), `vlm.py` (P1), `decision_engine.py` (P2), `semantic_sender.py` + `frame_sender.py` (P3), `main.py`
- `tcc/` [Mac 2] `pc2_receiver.py`, `camera_state.py` (P3), `dashboard/` (P4)
- `shared/schemas.py` (P3, contract) · `evaluation/` (fixed-rate baseline, metrics) · `tests/`
- `assets/demo_clips/` local only, never commit videos

## Rules for agents
1. **Stay in your lane:** edit only the module named in the task. Touching `shared/`, `docs/api-contract.md`, or this file → say so in the PR.
2. **No fake intelligence:** never hardcode labels, use filenames/ground truth to drive classification or sampling, or invent metrics. Mocks are allowed only behind an explicit `--mock` flag and must be labeled as mock in the UI.
3. **Reuse the existing paper code** (CLIP/OpenCLIP/SigLIP + APE) via a thin wrapper in `rsu/vlm.py`; map its class order explicitly. Do not re-derive APE.
4. **Never block inference** on the network: bounded queues, timeouts, drop stale JPEG previews, retry semantic messages with limits.
5. **Per-camera state is independent** (history, interval, state, priority, last alert). Add hysteresis/cooldown so rates don't flap.
6. Thresholds (0.60, 3 stable frames) are starting heuristics — keep them in one config, not scattered.
7. Small diffs, one feature per branch (`feature/<area>` or `<name>/<feature>`), PR into `main`. Add or update tests for logic.
8. Secrets only from env vars. Stack: Python 3.11+, OpenCV, FastAPI/uvicorn; dashboard per team choice.
