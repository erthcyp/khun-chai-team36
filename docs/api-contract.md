# API contract — RoadSense (draft, agree at kickoff)

Source: `docs/master-plan.md` §8. Change only via PR + tell the team.

Class order: `["normal_flow", "congestion", "accident", "invalid_scene"]`

## `POST /api/v1/semantic` (Mac 1 → Mac 2)
```json
{
  "schema_version": "1.0",
  "message_type": "semantic_update",
  "camera_id": "CAM_02",
  "location": {"label": "Example Junction", "latitude": null, "longitude": null},
  "frame_id": "CAM_02-000074",
  "video_timestamp_sec": 74.0,
  "captured_at": "2026-11-21T10:42:15+07:00",
  "predicted_class": "congestion",
  "score_order": ["normal_flow", "congestion", "accident", "invalid_scene"],
  "scores": [0.05, 0.84, 0.08, 0.03],
  "confidence": 0.84,
  "inference_ms": 85.2,
  "sampling_interval_sec": 1.0,
  "verification_state": "verifying",
  "alert": {"level": "preliminary", "type": "congestion", "message": "Congestion suspected"},
  "priority": "high"
}
```
Values are examples only. `verification_state`: `normal | verifying | alert | recovery | invalid`.
`alert`: `null` or `{level: preliminary|confirmed, type, message}`. `priority`: `low | medium | high | critical`.

## `POST /api/v1/snapshot` (async JPEG preview)
multipart: `file` (image/jpeg) + `meta` JSON `{camera_id, frame_id, video_timestamp_sec, event_id?}`

## Mac 2 read endpoints
`GET /health` · `GET /api/v1/cameras` · `GET /api/v1/alerts` · `GET /api/v1/metrics` · `WS /ws` (optional push)

## Open questions for 10:00 kickoff
- confidence = top softmax score? (must match what the existing VLM code returns)
- snapshot via HTTP multipart (above) or WebSocket?
- TCC port: 8000 conflicts with Docker on erthcyp's Mac → use 8010 if that Mac is Mac 2
