# RoadSense TCC dashboard

Responsive Thai-language TCC dashboard inspired by the supplied control-room reference. CAM_01–CAM_06 are available through the center camera selector; the selected camera becomes the primary feed, while another available camera appears alongside it. Every camera card contains its own natural-language explanation from the TCC meaning processor, or a waiting state until one arrives. Alerts stay in a right column on wide layouts. “แสดงกล้องทั้งหมด” replaces the main layout with an in-page 6-camera wall; it does not request browser fullscreen. The theme toggle switches between dark and light palettes and remembers the selection in the browser.

## Run

From the repository root, install and start the Mac 2 receiver:

```bash
python3 -m pip install -r tcc/requirements.txt
python3 -m uvicorn tcc.pc2_receiver:app --host 0.0.0.0 --port 8010
```

Open `http://localhost:8010/dashboard/` on Mac 2. Set the RSU sender's `TCC_URL` to `http://<Mac-2-LAN-IP>:8010`. The receiver accepts semantic JSON and JPEG snapshots on the endpoints in `docs/api-contract.md`, keeps the latest state per camera in memory, and serves the dashboard from the same origin. Restarting it clears the in-memory data.

The receiver uses `tcc/meaning_process.py`: it publishes camera state immediately and attaches a `meaning` object with `summary_th`, `summary_en`, `suggested_action`, `source`, `model`, and `generated_at` when the asynchronous result arrives. Set `roadsense_meaning` (or `OPENAI_API_KEY`) in the receiver process to enable image-backed OpenAI summaries. Without a key or matching JPEG, the meaning processor returns its labeled local fallback. Each camera card displays the summary and lets the operator switch between Thai and English.

For frontend-only preview, from the repository root:

```bash
python3 -m http.server 4173 --directory tcc/dashboard
```

Open `http://localhost:4173`. The page requests the TCC API from the same origin by default. To use another TCC host, append its origin:

```text
http://localhost:4173/?api=http%3A%2F%2F192.168.50.2%3A8000
```

The receiver must allow the dashboard origin through CORS when the static site and API use different origins. The page requests:

- `GET /health`
- `GET /api/v1/cameras`
- `GET /api/v1/alerts`

The layout always shows CAM_01 through CAM_06 as empty camera slots. Real records from `GET /api/v1/cameras` replace their matching slots by `camera_id`; the dashboard polls every three seconds, sorts them by camera number, and de-duplicates repeated IDs. Empty slots say "รอข้อมูล" and do not invent classifications, metrics, or descriptions. Records may include `latest_snapshot_url`, `snapshot_url`, `preview_url`, or `image_url` for the newest sampled JPEG. If none is present, the card shows a road-scene placeholder. Camera location labels come from the receiver payload; coordinates are shown when supplied.

## Mock preview

Append `?mock=1` to render six clearly labeled sample camera cards for previewing the layout. These mock records are used only in this explicit preview mode and are not used by the live API path. Remove `mock=1` for receiver-driven camera data.

## Notes

- The dashboard is read-only: it does not change RSU sampling intervals or classify images.
- Metrics are rendered from receiver counters. If no measured baseline or counters are available, the UI shows `—` instead of estimating them.
- Dark mode is the initial theme. Use the sun/moon control in the header to switch themes.
