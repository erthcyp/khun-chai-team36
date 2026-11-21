# AGENTS.md — apps/web
- Single `index.html`, served by the TCC at `/`. No build step unless the team agrees (keeps the demo setup trivial).
- Data sources only: `GET /api/events`, `GET /api/rsus`, `WS /ws` (messages `{type:"event"|"plan", data}`) — see `/docs/api.md`.
- Escape all server strings before inserting into HTML (use the existing `esc()`), agent output included.
- Ideas worth doing: map of RSUs, payload-size counter (16 B vs raw image), approve/reject buttons for plans that need human approval.
