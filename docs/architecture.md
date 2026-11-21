# Architecture

```mermaid
flowchart LR
  U[User] --> W[apps/web]
  W -->|REST, docs/api.md| A[apps/api]
  A --> G[Agent loop]
  G -->|tools| T[(Tools / data)]
  G --> M[OpenAI API]
```

## Components
- **web:** TODO
- **api:** TODO
- **agent:** TODO — tools it can call, how it handles failures/unexpected input (judges care about reliability)

## Responsible AI
- What data we use and where it goes: TODO
- Guardrails / human-in-the-loop: TODO
- Known limitations: TODO
