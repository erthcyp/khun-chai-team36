# AGENTS.md — instructions for Codex (and any coding agent)

Codex reads this file automatically. Keep it short and accurate; update it when the stack changes.
Sub-folders may have their own AGENTS.md (e.g. `apps/web/AGENTS.md`); the closest one wins.

## Project
- **Name:** TODO
- **One-liner:** TODO — what problem, for whom
- **Hackathon track:** TODO (Autonomous & Adaptive AI / AI-Native Products & Operations / Deep Domain AI)
- **Demo deadline:** TODO (today, before 21:00)

## Repo layout
- `apps/web/` — frontend (TODO: framework)
- `apps/api/` — backend / agent logic (TODO: framework)
- `docs/` — problem statement, architecture, pitch notes
- `scripts/` — dev helpers

## Commands (keep these working — Codex runs them to verify its work)
- Install: `make install`
- Run locally: `make dev`
- Lint + test: `make check`   ← run this before finishing any task

## Rules for agents
1. **Stay in your lane.** Only edit files inside the area named in the task (e.g. `apps/api/`). If you must touch shared files (`AGENTS.md`, `Makefile`, `.env.example`, API contracts in `docs/api.md`), say so explicitly in the PR description.
2. **Small diffs.** One feature per branch/PR. No drive-by refactors or reformatting of files you didn't need to change.
3. **Never commit secrets.** Read keys from env vars only; add new vars to `.env.example` with a dummy value.
4. **API contract first.** If you change a request/response shape, update `docs/api.md` in the same PR.
5. **Demo > perfection.** Prefer the simplest thing that works end-to-end. Hard-code/mock rather than block; mark it with `// TODO(hack):`.
6. **Run `make check`** before finishing. If something can't pass, explain why in the PR.
7. Write PR descriptions with: what changed, how to test, screenshots if UI.

## Conventions
- Branches: `<name>/<short-feature>` e.g. `nerdeye/agent-loop`
- Commits: imperative, short (`add retry to tool calls`)
- Language in code & comments: English
