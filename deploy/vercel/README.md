# RoadSense deployment and demo checklist

## Two-Mac live demo

- [ ] Check out the same tested integration commit on both Macs. Run the test suite and install the dependencies for each role. See [RSU baseline](../../docs/rsu-baseline.md) for its setup.
- [ ] On Mac 1, stage both clips and the model weights locally. Verify the camera IDs, clip paths, and location metadata in the RSU camera config once it lands. Warm up real inference before presenting.
- [ ] Connect both Macs over a tested Thunderbolt Bridge or private Wi-Fi. Record the actual Mac 2 address. Keep the second link available as a fallback.
- [ ] On Mac 2, start the TCC receiver on the private link. Use port 8010 if 8000 is occupied. Start the dashboard with the local TCC API as its data source. Set the OpenAI key on Mac 2 if summaries are enabled and verify the template fallback.
- [ ] On Mac 1, point `TCC_URL` at the Mac 2 address and port. Run `curl --fail --show-error "${TCC_URL}/health"` before starting the RSU, then check that both cameras update on the Mac 2 dashboard.
- [ ] Rehearse with real clips. Verify one camera goes from 2 s to 1 s and back while the other stays at 2 s. Check alerts, matching snapshots if enabled, and the measured inference-call comparison against fixed 1 s sampling on the same clips and duration.
- [ ] Rehearse a receiver restart. Keep the local dashboard open on the projector and have a labeled recording available for recovery.

Use the eventual integration start commands for the two processes. This scaffold has no dependency on either process.

## Vercel scaffold

- [ ] Import the repository into Vercel and set the project Root Directory to `deploy/vercel`.
- [ ] Use the Other framework preset and the checked-in output directory `public`. The placeholder has no install or build step.
- [ ] Open the deployment URL in a signed-out browser on a separate network and verify the placeholder loads. Confirm judge access before sharing the URL.

## Judge-facing dashboard

- [ ] Replace the placeholder with the dashboard build. If the team uses Vite, set the build command and output directory to match that app. Verify reloads on any client-side routes.
- [ ] Choose the data source. For a live link, publish selected camera, alert, and metric updates from Mac 2 to an authenticated HTTPS service with shared storage. Serve only the intended read endpoints to viewers. Keep ingestion credentials and the OpenAI key server-side. For a replay link, deploy a recorded run that is safe to share and label it clearly.
- [ ] Set the frontend API base URL. Use a same-origin API route or configure the external API to allow the dashboard origin through CORS. Start with polling for live updates and display the last update time so stale data is visible.
- [ ] Test from a judge's device or mobile data. Verify the link opens, updates appear, viewer access cannot write updates, and the local demo keeps running if the cloud link drops.

## Run Mac 2 from one terminal

Stop the existing TCC in its terminal before starting the launcher. From the repository root with the Mac 2 virtual environment active:

```bash
set -a
source .env
set +a
python deploy/vercel/run_mac2.py --host 172.20.10.11
```

Use Mac 2's current LAN IP for `--host`. The launcher starts the receiver on port 8010 and the read-only proxy on loopback port 8012. Pass `--cloudflared /path/to/cloudflared` if it is outside your PATH. Pressing Ctrl+C in this terminal closes the receiver, proxy and tunnel. The Vercel page still loads while the data feed is offline.

Quick Tunnel gives you a new URL each time. To reuse the same upstream URL after a restart, configure an existing Cloudflare named tunnel with a public hostname and a local service of `http://127.0.0.1:8012`. Point the Vercel rewrites at that hostname once and pass `--tunnel-config /path/to/config.yml --tunnel-name NAME` to the launcher.
