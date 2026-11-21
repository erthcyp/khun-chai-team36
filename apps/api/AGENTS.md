# AGENTS.md — apps/api
- Backend + agent logic. Implement endpoints exactly as specified in `/docs/api.md`.
- OpenAI key from `OPENAI_API_KEY` env var only. Model name from `OPENAI_MODEL`.
- Agent code lives in `agent/`. Every tool call: timeout, retry once, and a graceful fallback message — reliability is a judging angle.
- Log agent steps (tool name, input summary, result status) so the demo can show what the agent did.
- TODO: framework (FastAPI / Express), how to run tests.
