# RoadSense TCC dashboard

Static, responsive operator dashboard for the TCC receiver. It polls the documented JSON endpoints every three seconds and keeps camera state, alerts, metrics, and the assistant summary visible as separate panels.

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
- `GET /api/v1/summary` (optional until the assistant endpoint is implemented)

Camera records may include `latest_snapshot_url`, `snapshot_url`, `preview_url`, or `image_url` for the newest sampled JPEG. If none is present, the card shows a waiting state. The dashboard displays the incoming `score_kind` and uses the message `score_order` only for any future vector breakdown; it does not assume another class ordering.

## Mock preview

Append `?mock=1` to render clearly labeled sample UI data. Mock values are hard-coded only in this explicit preview mode and are not used by the live API path. Remove `mock=1` for real receiver data.

## Notes

- The dashboard is read-only: it does not change RSU sampling intervals or classify images.
- Metrics are rendered from receiver counters. If no measured baseline or counters are available, the UI shows `—`/waiting copy instead of estimating them.
- The class order in the current repository contract remains `normal_flow, congestion, accident, invalid_scene`.
