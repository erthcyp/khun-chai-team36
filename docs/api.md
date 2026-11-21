# Contracts

## 1. RSU → TCC: semantic payload
`POST /ingest`
- Body: **exactly 16 bytes** = 4 × float32 little-endian `[normal_flow, congestion, accident, invalid_scene]`, sum ≈ 1 (paper Eq. 6). Codec: `semcom/payload.py`.
- Headers:
  | Header | Example | Meaning |
  |---|---|---|
  | `X-RSU-ID` | `RSU-01` | sender id |
  | `X-Seq` | `42` | sequence number |
  | `X-Send-Ms` | `1763700000000` | unix ms at send |
  | `X-CPV` | `skip` / `checked` / `swapped` | CPV outcome |
  | `X-Frame` | `img_0012.jpg` | optional, debug |
- Response: `204` ok, `400` bad payload.

## 2. TCC → dashboard
- `GET /health` → `{"status":"ok"}`
- `GET /api/events?limit=50` → `Event[]`
- `GET /api/rsus` → `{ [rsu_id]: {state, confidence, last_seen} }`
- `WS /ws` → messages:
  - `{"type":"event","data":Event}`
  - `{"type":"plan","data":{"rsu_id","seq","plan":Plan}}`

```jsonc
// Event
{ "rsu_id":"RSU-01", "seq":7, "state":"accident", "confidence":0.449,
  "vector":{"normal_flow":0.392,"congestion":0.089,"accident":0.449,"invalid_scene":0.07},
  "cpv":"swapped", "frame":"mock", "payload_bytes":16, "recv_ms":1763700000123, "clock_delta_ms":3 }

// Plan (see PLAN_SCHEMA in apps/tcc/agent.py)
{ "severity":"high", "summary":"...", "actions":[{"type":"dispatch_rescue","detail":"..."}],
  "public_alert_th":"...", "public_alert_en":"...", "needs_human_approval":true, "source":"openai|fallback" }
```

The agent runs only when an RSU's state **changes** into `congestion`, `accident` or `invalid_scene`.
