# Codex usage log

Track what Codex built — useful for the pitch ("built with Codex") and for watching the $100 credit.

| Time | Who | Task / prompt (short) | Result (PR #) | Notes |
|---|---|---|---|---|
| | | | | |
| 2026-11-21 | kantheedeth | Read README; build configurable OpenCV frame automation, default 1 frame/s | Branch `kantheedeth/frame-automation` | Per-camera sampler, runtime interval API, CLI, generated-video tests; no classification or fabricated metrics |
| 21 Nov 2026 | Codex | Import the selected RSU baseline into the team repo | `feature/rsu-baseline` | Reused APE K=30, S-BERT selector and OpenAI ViT-L/14. Adapted verification and transport to the team contract. |
| 2026-11-21 | kantheedeth | Combine sampler and VLM baseline for integration testing; install dependencies | Branch `kantheedeth/frame-vlm-integration` | Preserved both source branches; merged existing K30 prompts and VLM modules; inference wiring left for the user |
| 2026-11-21 | kantheedeth | Diagnose QuickGELU mismatch during video inference | Branch `kantheedeth/frame-vlm-integration` | Explicitly enable QuickGELU to match OpenAI pretrained weights; preserve the model, K30 prompts and class order |
| 2026-11-21 | kantheedeth | Add `rsu/main.py` video selection and real VLM throughput runner | Branch `kantheedeth/frame-vlm-integration` | 28 tests pass; five real frames from video48 on MPS: 1.04 outputs/s, mean prediction 166 ms, zero 1-second processing overruns; startup excluded; single-camera smoke test only |
