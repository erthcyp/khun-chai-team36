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
  "priority": "high",
  "event_id": "CAM_02-evt-0003"
}
```
Values are examples only. `verification_state`: `normal | verifying | alert | recovery | invalid`.
`alert`: `null` or `{level: preliminary|confirmed, type, message}`. `priority`: `low | medium | high | critical`.

## `POST /api/v1/snapshot` (async JPEG preview)
multipart: `file` (image/jpeg) + `meta` JSON `{camera_id, frame_id, video_timestamp_sec, event_id?}`

## Mac 2 read endpoints
`GET /health` · `GET /api/v1/cameras` · `GET /api/v1/alerts` · `GET /api/v1/metrics` · `WS /ws` (optional push)

## Meaning process (Mac 2, OpenAI)
- Runs on the TCC only; key from `OPENAI_API_KEY` env on Mac 2.
- **Trigger:** only when a camera's `verification_state`, `predicted_class` or `alert.level` changes (or a new `event_id` appears) — never on every message.
- **Inputs:** semantic JSON plus a JPEG with the same `camera_id` and `frame_id`. When the JPEG supplies an `event_id`, it must match the semantic update.
- **Timing:** publish camera status and alerts immediately. The asynchronous summary worker waits up to 2 seconds for the matching JPEG by default. It calls OpenAI once per qualifying update when both inputs are available. A late or duplicate JPEG does not trigger another call.
- Write Thai and English summaries. Include `location.label` and `captured_at` when supplied. Use the image to refine wording and keep factual claims within the semantic JSON.
- **Fallback:** use a local JSON template if the image times out, the key is unavailable, or the API fails. When supplied, `fallback_reason` is `image_timeout | missing_frame_id | api_unavailable | api_error`. The dashboard can show the image-timeout status alongside the template. No API call is made without a matching image.
- Output attached to the alert:
```json
{"event_id": "CAM_02-evt-0003", "summary_en": "...", "summary_th": "...",
 "suggested_action": "...", "source": "json+image | fallback", "model": "...", "generated_at": "..."}
```
Exposed via `GET /api/v1/alerts` and pushed on `WS /ws` as `{"type": "meaning", "data": {...}}`.

## Decisions (kickoff, 10:47)
- **Transport: option A, JSON** (this file). Binary 16-byte vector is a stretch goal only.
- **Scores:** `scores` = softmax(100 × cosine(image, APE class embedding)) per paper Eq. 5–6, in `score_order`; sums to 1.
- **Confidence:** `confidence = max(scores)`. Thresholds (e.g. 0.60) refer to this value.
- **Location comes from config, not the model:** RSU loads `rsu/cameras.json` and copies the camera's `location` into every message.
- **Measure bytes:** TCC records the size of every semantic message and every JPEG so the dashboard can show real bandwidth vs raw frames.

### `rsu/cameras.json` (example)
```json
[
  {"camera_id": "CAM_01", "video": "assets/demo_clips/cam01.mp4",
   "location": {"label": "Junction A", "latitude": null, "longitude": null}},
  {"camera_id": "CAM_02", "video": "assets/demo_clips/cam02.mp4",
   "location": {"label": "Junction B", "latitude": null, "longitude": null}}
]
```

## Still open
- Model: CLIP ViT-L-14 (best in paper) vs ViT-B-32 (faster) — decide after timing `predict()` on Mac 1
- Prompt set: the paper's 120 APE prompts (30/class) — confirm file location
- Snapshot via HTTP multipart (above) or WebSocket?
- TCC port: 8000 conflicts with Docker on erthcyp's Mac → use 8010 if that Mac is Mac 2
