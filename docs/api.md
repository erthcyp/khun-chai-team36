# API contract (frontend ⇄ backend)

Agree on this early. Change only via PR + tell the team.

Base URL (local): `http://localhost:8000`

## `GET /health`
Response `200`: `{ "status": "ok" }`

## TODO: main endpoint
`POST /api/TODO`

Request:
```json
{ "TODO": "string" }
```

Response `200`:
```json
{ "TODO": "string" }
```

Errors: `{ "error": "message" }` with 4xx/5xx.
