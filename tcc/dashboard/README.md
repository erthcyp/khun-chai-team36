# RoadSense TCC dashboard

Responsive Thai-language TCC dashboard inspired by the supplied control-room reference. It shows camera cards, an incident panel, and a measured adaptive-versus-fixed model-call comparison. The theme toggle switches between dark and light palettes and remembers the selection in the browser.

## Run

From the repository root:

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
- `GET /api/v1/metrics`

Camera records for `CAM_01` through `CAM_06` are discovered from `GET /api/v1/cameras` and rendered as they appear; the dashboard polls every three seconds, sorts them by camera number, and de-duplicates repeated IDs. No missing cameras are fabricated. If offline cameras should remain visible, the receiver must return their registry records with an offline state. Records may include `latest_snapshot_url`, `snapshot_url`, `preview_url`, or `image_url` for the newest sampled JPEG. If none is present, the card shows a road-scene placeholder. Camera location labels come from the receiver payload; coordinates are shown when supplied.

## Mock preview

Append `?mock=1` to render six clearly labeled sample camera cards for previewing the layout. These mock records are used only in this explicit preview mode and are not used by the live API path. Remove `mock=1` for receiver-driven camera data.

## Notes

- The dashboard is read-only: it does not change RSU sampling intervals or classify images.
- Metrics are rendered from receiver counters. If no measured baseline or counters are available, the UI shows `—` instead of estimating them.
- Dark mode is the initial theme. Use the sun/moon control in the header to switch themes.
