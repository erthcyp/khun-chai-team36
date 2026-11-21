# Team workflow — RoadSense

## Roles (master-plan §9.2 — fill in names at kickoff)
| Role | Owner | Owns | Deliverable |
|---|---|---|---|
| P1 VLM / Video | TODO | `rsu/vlm.py`, `rsu/frame_automation.py` | tested `predict(frame)` + per-camera frames |
| P2 Adaptive decision | TODO | `rsu/decision_engine.py` | unit-tested state machine, 2 s ↔ 1 s, alerts |
| P3 Network / backend | TODO | `shared/schemas.py`, `rsu/*_sender.py`, `tcc/pc2_receiver.py`, `tcc/camera_state.py` | JSON + JPEG received on the real second Mac |
| P4 Dashboard / integration / pitch | TODO | `tcc/dashboard/`, `evaluation/`, demo script | dashboard + repeatable 2–3 min demo |

One owner per interface, one integration lead who merges PRs.

## Setup
```bash
git clone https://github.com/erthcyp/khun-chai-team36.git && cd khun-chai-team36
python3 -m venv .venv && source .venv/bin/activate
```
Open the folder as a project in the Codex app (it reads `AGENTS.md`).

## Loop
```bash
git checkout main && git pull
git checkout -b feature/<area>
# ask Codex for ONE scoped change in your module, e.g.:
# "In rsu/decision_engine.py only: implement the per-camera state machine from docs/master-plan.md §7 with unit tests in tests/test_decision_engine.py"
# read the diff yourself, then:
git push -u origin HEAD   # open a PR; integration lead merges
```
Never push to `main` directly; never force-push. Log notable Codex tasks in `docs/codex-log.md` (judging criterion).

## Milestones (master-plan §14)
11:45–12:30 one camera end-to-end · 13:30–14:30 adaptive 2 s↔1 s works · 15:15–16:00 demo-ready · **17:00 code freeze & submit**
