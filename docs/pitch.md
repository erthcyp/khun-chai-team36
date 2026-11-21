# Pitch notes (target: 3–5 min)

1. **Hook** (30s) — Bangkok CCTV: thousands of cameras, a few operators. An accident sits on screen #214 for minutes before anyone sees it.
2. **Solution** (30s) — the camera understands the scene, sends 16 bytes of meaning, the control center's agent drafts the response.
3. **Live demo** (2 min)
   - Dashboard with RSUs streaming normal flow
   - Congestion → agent suggests signal timing + sign message
   - Ambiguous frame → CPV partners catch it, flips to accident → plan needs operator approval
   - Point at the payload counter: 16 B vs ~100 KB per frame
4. **Why it's credible** (30s) — research numbers: 67% → 84% mean accuracy, 94.3% best; 40% lower latency, 3.7% vs 35.3% deadline misses.
5. **Built with Codex** (30s) — what Codex built today (codex-log.md).
6. **Responsible + next** (30s) — no raw images leave the roadside; human-in-the-loop for dispatch; next: cellular testing, Thai road data.

## Likely judge questions
- What was built today vs before? (Be upfront: research = APE/CPV; today = agent, TCC, dashboard, real-time pipeline.)
- What happens on a foggy/blocked camera? (invalid_scene → inspect_camera, no false accident.)
- Why VLM instead of a trained detector? (zero-shot, new classes by editing prompts, no labeled data.)
- Who buys this? (city traffic centers, expressway / toll operators.)
