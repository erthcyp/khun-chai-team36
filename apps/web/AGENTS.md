# AGENTS.md — apps/web
- Frontend only. Talk to the backend strictly via the endpoints in `/docs/api.md`.
- Read the API base URL from env (e.g. `VITE_API_URL` / `NEXT_PUBLIC_API_URL`), never hard-code.
- If the backend isn't ready, mock responses that match `/docs/api.md` exactly.
- TODO: framework, styling lib, component conventions.
