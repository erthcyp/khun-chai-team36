# AGENTS.md — apps/tcc
- FastAPI. Endpoints must match `/docs/api.md`.
- `POST /ingest` takes exactly 16 bytes (`semcom.payload.decode`); reject anything else with 400.
- The agent (`agent.py`) runs in a background task; ingest must stay fast and never wait on OpenAI.
- OpenAI: key from `OPENAI_API_KEY`, model from `OPENAI_MODEL`. Keep `_fallback()` working — it is the stage-safety net.
- Agent output must validate against `PLAN_SCHEMA`; if you add fields, update the schema and the dashboard together.
