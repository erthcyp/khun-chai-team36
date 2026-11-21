# AGENTS.md — instructions for Codex (and any coding agent)

Codex reads this file automatically. Sub-folders have their own AGENTS.md; the closest one wins.

## Project
- **What:** RSU-edge traffic-state inference with VLMs → 16-byte semantic vector → Traffic Control Center (TCC) → OpenAI agent that drafts an incident response plan → live dashboard.
- **Based on:** our paper "Adaptive Prompt Ensemble for VLM-Based Traffic State Inference in RSU Semantic Communication Systems" (APE + CPV). The hackathon adds the TCC response agent (paper's future work) and the product demo.
- **Tracks:** Deep Domain AI (ITS / traffic) + Autonomous & Adaptive AI (CPV self-checks uncertain predictions; agent escalates to humans when unsure).
- **Classes (fixed order, never change):** `normal_flow, congestion, accident, invalid_scene`.

## Repo layout
- `semcom/` — shared: `payload.py` (16-byte codec), `cpv.py` (Algorithm 1). Pure numpy. **Contract code — change only with team agreement.**
- `apps/rsu/` — edge sender. `engine.py` has `MockEngine` (no ML) and `ClipEngine` (open_clip, APE Eq. 5–6, CPV partners).
- `apps/tcc/` — FastAPI receiver (`main.py`) + response agent (`agent.py`, OpenAI Responses API with JSON schema + rule fallback).
- `apps/web/` — single-file dashboard served by the TCC at `/` (no build step).
- `prompts/ape_prompts.json` — APE prompts per class.
- `tests/` — pytest.

## Commands
- Install: `make install` (light) / `make install-clip` (real VLMs)
- Run demo: `make dev` → TCC + 2 mock RSUs, open http://localhost:8000
- Lint + test: `make check` ← must pass before finishing any task

## Rules for agents
1. **Stay in your lane.** Only edit files in the area named in the task. Shared files (`semcom/`, `AGENTS.md`, `Makefile`, `docs/api.md`, `.env.example`) → call it out in the PR description.
2. **The payload stays 16 bytes.** Metadata goes in HTTP headers, never into the payload. This is the paper's core claim.
3. **Small diffs**, one feature per PR, no drive-by refactors.
4. **Never commit secrets.** Keys from env only; add new vars to `.env.example`.
5. **The demo must never crash.** Every external call (OpenAI, model download, network) needs a timeout and a fallback. Mock mode must keep working.
6. **Agent safety:** `dispatch_rescue` always requires human approval; low confidence or CPV-changed labels → `needs_human_approval=true`.
7. Add/adjust a test in `tests/` for logic changes. Run `make check`.
8. PR description: what changed, how to test, screenshot if UI.

## Conventions
- Python 3.11+, type hints, `ruff` (config in `ruff.toml`).
- Branches `<name>/<feature>`; imperative commit messages; English in code.
