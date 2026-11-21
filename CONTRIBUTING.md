# Team workflow (one-day hackathon)

Goal: several people + several Codex agents working at once **without merge hell**.

## 0. Setup (everyone, first 15 min)
1. Accept the GitHub invite, then `git clone <repo-url> && cd <repo>`
2. `cp .env.example .env` and paste the shared key (get it from the lead in person/DM — never in the repo or group chat screenshots)
3. Install Codex: `npm i -g @openai/codex` then `codex` and sign in.
   Or use the Codex extension in VS Code / Cursor, or Codex cloud at chatgpt.com/codex (connect the GitHub repo once).
4. `make install && make dev` → open http://localhost:8000 and confirm events stream in.

## 1. Ownership (avoid conflicts)
Each person owns an area. Codex tasks you start must stay inside your area (see AGENTS.md rule 1).

4 people, 4 lanes:

| # | Area | Owner | Folders | First tasks |
|---|---|---|---|---|
| 1 | RSU / VLM | TODO | `apps/rsu/`, `prompts/` | Load the real 120 APE prompts, run `ClipEngine` on sample images from the paper dataset, pick demo frames |
| 2 | TCC + agent | TODO | `apps/tcc/` | Real OpenAI key, tune agent prompt, add operator approve/reject endpoint |
| 3 | Dashboard | TODO | `apps/web/` | Map/grid of RSUs, payload-size counter, approve/reject buttons |
| 4 | Integration + pitch (lead) | Bhuchi | `Makefile`, `.github/`, `docs/`, `README.md`, `semcom/` | Merge PRs, keep `main` green, demo script, backup video, slides |

Shared contracts: `docs/api.md` and `semcom/`. Change only via PR + tell the group.

## 2. Loop
```bash
git checkout main && git pull
git checkout -b <name>/<feature>
codex "in apps/web only: add approve/reject buttons to plans that need human approval, per docs/api.md; run make check"
# review the diff yourself — don't merge what you haven't read
git push -u origin HEAD   # open a PR
```
- Running 2+ Codex tasks yourself? Use one git worktree per task:
  `git worktree add ../wt-<feature> -b <name>/<feature>`
- Codex cloud tasks open their own branches/PRs — still one feature per task.

## 3. Merging
- PRs small, merge often (aim every 30–60 min). Integration owner merges.
- Ask Codex to review: comment `@codex review` on the PR (if the GitHub integration is enabled).
- `main` must always run. If it breaks, fixing it beats everything else.

## 4. Timeline (suggested)
| Time | Milestone |
|---|---|
| ~10:00 | Idea locked, `docs/api.md` agreed, skeleton runs |
| ~13:00 | End-to-end happy path works (ugly is fine) |
| ~17:00 | Feature freeze. Only polish & bug fixes |
| ~18:30 | Deployed demo + backup video recorded |
| ~19:30 | Pitch rehearsed twice |

## 5. Credit budget
Shared $100 Codex credit. Log big runs in `docs/codex-log.md`. Prefer specific, scoped prompts over "build the whole app".
