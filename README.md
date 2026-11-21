# TODO: Product name — Semantic Traffic Intelligence at the Roadside

> Sea x OpenAI Regional Codex Hackathon (Thailand) — 21 Nov 2026
> Team: **khun-chai (TEAM36)**
> Tracks: Deep Domain AI · Autonomous & Adaptive AI

## Problem
Traffic CCTV streams raw video to control centers: heavy bandwidth, high latency, privacy risk, and operators drowning in feeds. Incidents are noticed late.

## Solution
Each roadside unit understands the scene itself with vision-language models (APE prompts + CPV self-verification), sends only a **16-byte** meaning vector, and the control center's OpenAI agent drafts the response plan — with a human approving anything risky.

Built on our research (APE + CPV, ICOIN 2027 submission): 84.39% mean accuracy across 9 VLMs (best 94.30%), 40.65% lower latency than raw images.

## Quick start
```bash
cp .env.example .env      # OPENAI_API_KEY optional — rule-based fallback without it
make install
make dev                  # TCC + 2 mock RSUs → http://localhost:8000
make check                # lint + tests
```
Real VLMs at the RSU: `make install-clip`, put images in `data/samples/`, `make rsu-clip`.

## Docs
- [Architecture](docs/architecture.md) · [Contracts](docs/api.md) · [Pitch](docs/pitch.md) · [Codex log](docs/codex-log.md)

## Team — khun-chai (TEAM36)
| Name | Role |
|---|---|
| Bhuchi Kumprakob | TODO |
| Chaiyapat Anantarasuchart | TODO |
| Chaitawat Boonkitticharoen | TODO |
| Kanteeded Limkangwalmongkol | TODO |
