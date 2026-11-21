(() => {
  "use strict";

  const REFRESH_MS = 3000;
  const STALE_AFTER_MS = 12000;
  const CLASS_ORDER = ["normal_flow", "congestion", "accident", "invalid_scene"];
  const mockMode = new URLSearchParams(location.search).get("mock") === "1";
  const apiBase = (new URLSearchParams(location.search).get("api") || "").replace(/\/$/, "");
  const state = { cameras: [], alerts: [], metrics: null, summary: null, healthy: false, lastSuccess: null, lastMetricSuccess: null, lastSummarySuccess: null };
  const $ = (id) => document.getElementById(id);

  if (mockMode) document.body.classList.add("mock-mode");

  function esc(value) {
    return String(value ?? "");
  }
  function num(value) {
    if (value === null || value === undefined || value === "") return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  function first(obj, keys) {
    for (const key of keys) if (obj && obj[key] !== undefined && obj[key] !== null) return obj[key];
    return null;
  }
  function arrayFrom(payload, keys) {
    if (Array.isArray(payload)) return payload;
    for (const key of keys) if (Array.isArray(payload?.[key])) return payload[key];
    return [];
  }
  function fmtTime(value) {
    if (!value) return "—";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? "—" : date.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit", timeZone: "Asia/Bangkok" });
  }
  function relative(value) {
    if (!value) return "No sample time";
    const t = new Date(value).getTime();
    if (!Number.isFinite(t)) return "Time unavailable";
    const sec = Math.max(0, Math.floor((Date.now() - t) / 1000));
    if (sec < 2) return "just now";
    if (sec < 60) return `${sec}s ago`;
    const min = Math.floor(sec / 60);
    return `${min}m ago`;
  }
  function fmtBytes(bytes) {
    const n = num(bytes);
    if (n === null) return "—";
    if (n < 1000) return `${Math.round(n)} B`;
    if (n < 1_000_000) return `${(n / 1000).toFixed(1)} KB`;
    return `${(n / 1_000_000).toFixed(2)} MB`;
  }
  function fmtPct(value) {
    const n = num(value);
    return n === null ? "—" : `${n.toFixed(1)}%`;
  }
  function normalizedClass(value) {
    const label = String(value || "unknown").toLowerCase().replace(/[ -]/g, "_");
    return CLASS_ORDER.includes(label) ? label : label;
  }
  function displayClass(value) {
    const label = normalizedClass(value);
    return label === "unknown" ? "Awaiting result" : label.replaceAll("_", " ");
  }
  function locationText(camera) {
    const loc = camera.location || {};
    const label = first(loc, ["label", "name"]) || camera.location_label || "Location not provided";
    const lat = num(first(loc, ["latitude", "lat"]));
    const lon = num(first(loc, ["longitude", "lon", "lng"]));
    return { label: esc(label), coordinates: lat !== null && lon !== null ? `${lat.toFixed(4)}, ${lon.toFixed(4)}` : "Coordinates unavailable" };
  }
  function sampleTime(camera) {
    return first(camera, ["captured_at", "updated_at", "received_at", "last_seen_at", "timestamp"]);
  }
  function currentPrediction(camera) {
    return first(camera, ["predicted_class", "prediction", "final_class", "classification"]);
  }
  function currentScore(camera) {
    let score = first(camera, ["local_top_score", "confidence", "top_score"]);
    if (score === null && Array.isArray(camera.scores || camera.semantic_vector)) {
      const scores = camera.scores || camera.semantic_vector;
      score = Math.max(...scores.map(Number).filter(Number.isFinite));
    }
    return num(score);
  }
  function isOnline(camera) {
    const explicit = first(camera, ["online", "is_online", "connected"]);
    if (explicit !== null) return Boolean(explicit);
    const at = sampleTime(camera);
    return at ? Date.now() - new Date(at).getTime() <= STALE_AFTER_MS : false;
  }
  function priority(camera) {
    const p = String(camera.priority || "low").toLowerCase();
    return ["critical", "high", "medium", "low"].includes(p) ? p : "low";
  }
  function scoreLabel(camera) {
    const score = currentScore(camera);
    const kind = camera.score_kind || camera.score_type || (camera.confidence !== undefined ? "confidence" : "local score");
    return { value: score === null ? "—" : (score <= 1 ? `${(score * 100).toFixed(0)}%` : score.toFixed(2)), kind: esc(String(kind).replaceAll("_", " ")) };
  }

  async function getJson(path) {
    const response = await fetch(`${apiBase}${path}`, { cache: "no-store", headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(`${path} returned ${response.status}`);
    return response.json();
  }
  async function poll() {
    if (mockMode) {
      state.cameras = mockData.cameras;
      state.alerts = mockData.alerts;
      state.metrics = mockData.metrics;
      state.summary = mockData.summary;
      state.healthy = true;
      state.lastSuccess = new Date();
      state.lastMetricSuccess = new Date();
      state.lastSummarySuccess = new Date();
      render();
      return;
    }
    const [health, cameras, alerts, metrics, summary] = await Promise.allSettled([
      getJson("/health"), getJson("/api/v1/cameras"), getJson("/api/v1/alerts"), getJson("/api/v1/metrics"), getJson("/api/v1/summary")
    ]);
    const healthOk = health.status === "fulfilled";
    const camerasOk = cameras.status === "fulfilled";
    const alertsOk = alerts.status === "fulfilled";
    const metricsOk = metrics.status === "fulfilled";
    const summaryOk = summary.status === "fulfilled";
    state.healthy = healthOk && (camerasOk || alertsOk || metricsOk);
    if (camerasOk) state.cameras = arrayFrom(cameras.value, ["cameras", "items", "data"]);
    if (alertsOk) state.alerts = arrayFrom(alerts.value, ["alerts", "events", "items", "data"]);
    if (metricsOk) { state.metrics = metrics.value?.metrics || metrics.value; state.lastMetricSuccess = new Date(); }
    if (summaryOk) { state.summary = summary.value?.summary || summary.value; state.lastSummarySuccess = new Date(); }
    if (state.healthy) state.lastSuccess = new Date();
    render();
  }

  function cameraSort(a, b) {
    const order = { critical: 0, high: 1, medium: 2, low: 3 };
    const pa = order[priority(a)] ?? 4;
    const pb = order[priority(b)] ?? 4;
    if (pa !== pb) return pa - pb;
    const alertA = a.alert ? 0 : 1, alertB = b.alert ? 0 : 1;
    if (alertA !== alertB) return alertA - alertB;
    return String(a.camera_id || a.id || "").localeCompare(String(b.camera_id || b.id || ""));
  }
  function cameraCard(camera) {
    const id = esc(camera.camera_id || camera.id || "Unknown camera");
    const loc = locationText(camera);
    const cls = normalizedClass(currentPrediction(camera));
    const score = scoreLabel(camera);
    const online = isOnline(camera);
    const status = String(camera.verification_state || camera.state || (online ? "normal" : "offline")).toLowerCase();
    const interval = num(camera.sampling_interval_sec);
    const preview = first(camera, ["latest_snapshot_url", "snapshot_url", "preview_url", "image_url"]);
    const time = sampleTime(camera);
    const pr = priority(camera);
    const prClass = pr === "low" ? "low" : pr;
    const isStale = !online;
    const article = document.createElement("article");
    article.className = `camera-card priority-${pr}`;
    const head = document.createElement("div"); head.className = "camera-card-head";
    const left = document.createElement("div"); left.className = "camera-id-group";
    left.innerHTML = `<span class="camera-live-dot ${online ? "online" : ""} ${isStale ? "stale" : ""}"></span><span><span class="camera-name">${escapeHtml(id)}</span><span class="camera-location">${escapeHtml(loc.label)}</span></span>`;
    const badge = document.createElement("span"); badge.className = `priority-badge ${prClass}`; badge.textContent = `${pr.toUpperCase()} PRIORITY`;
    head.append(left, badge);
    const previewEl = document.createElement("div"); previewEl.className = "preview";
    if (preview) {
      const img = document.createElement("img"); img.src = String(preview); img.alt = `Latest sampled snapshot from ${id}`; img.loading = "lazy";
      img.onerror = () => { img.remove(); if (!previewEl.querySelector(".preview-empty")) previewEl.insertAdjacentHTML("afterbegin", '<div class="preview-empty"><span class="preview-road"></span></div>'); };
      previewEl.appendChild(img);
    } else {
      const placeholder = document.createElement("div"); placeholder.className = "preview-empty"; placeholder.innerHTML = '<span class="preview-road"></span>'; previewEl.appendChild(placeholder);
    }
    const previewStatus = document.createElement("span"); previewStatus.className = `camera-status ${isStale ? "stale" : ""}`; previewStatus.textContent = online ? "LIVE FEED" : "STALE FEED";
    const frameLabel = document.createElement("span"); frameLabel.className = "preview-label"; frameLabel.textContent = preview ? "LATEST SAMPLED SNAPSHOT" : "WAITING FOR SNAPSHOT";
    const fresh = document.createElement("span"); fresh.className = "freshness"; fresh.textContent = relative(time);
    previewEl.append(previewStatus, frameLabel, fresh);
    const data = document.createElement("div"); data.className = "camera-data";
    const classRow = document.createElement("div"); classRow.className = "classification-row";
    const label = document.createElement("div"); label.className = `class-label ${cls}`; label.innerHTML = `<span class="class-icon">${classIcon(cls)}</span><span>${escapeHtml(displayClass(cls))}</span>`;
    const scoreInfo = document.createElement("div"); scoreInfo.className = "score-info"; scoreInfo.innerHTML = `<span>${escapeHtml(score.value)}</span><small>${escapeHtml(score.kind)}</small>`;
    classRow.append(label, scoreInfo);
    const metaRow = document.createElement("div"); metaRow.className = "camera-meta-row";
    const v = document.createElement("span"); v.className = `verification ${statusClass(status)}`; v.innerHTML = `<span class="v-dot"></span>${escapeHtml(status.replaceAll("_", " ").toUpperCase())}`;
    const sample = document.createElement("span"); sample.className = "sampling"; sample.innerHTML = `SAMPLE <strong>${interval === null ? "—" : `${interval.toFixed(interval % 1 ? 1 : 0)}s`}</strong>`;
    metaRow.append(v, sample);
    const locationRow = document.createElement("div"); locationRow.className = "location-line";
    locationRow.innerHTML = `<span>${escapeHtml(loc.label)}</span><span>${escapeHtml(loc.coordinates)}</span>`;
    data.append(classRow, metaRow, locationRow);
    article.append(head, previewEl, data);
    return article;
  }
  function escapeHtml(str) {
    return esc(str).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
  }
  function classIcon(cls) { return cls === "accident" ? "⚠" : cls === "congestion" ? "≋" : cls === "invalid_scene" ? "◌" : "↗"; }
  function statusClass(status) { return /verify|pending/.test(status) ? "verify" : /alert|accident|confirm/.test(status) ? "alert" : ""; }

  function renderCameras() {
    const container = $("camera-list");
    const cams = [...state.cameras].sort(cameraSort);
    $("nav-camera-count").textContent = String(cams.length || 0);
    $("camera-section-count").textContent = `${cams.length} FEED${cams.length === 1 ? "" : "S"}`;
    if (!cams.length) {
      container.innerHTML = `<div class="empty-state"><span class="${state.healthy ? "" : "empty-spinner"}"></span><strong>${state.healthy ? "No camera data available" : "TCC receiver is unavailable"}</strong><span>${state.healthy ? "The receiver is online but has not reported any cameras." : "Start the TCC API, then refresh to load live camera state."}</span></div>`;
      return;
    }
    container.replaceChildren(...cams.map(cameraCard));
  }
  function renderAlerts() {
    const alerts = [...state.alerts].sort((a, b) => new Date(first(b, ["captured_at", "created_at", "timestamp"]) || 0) - new Date(first(a, ["captured_at", "created_at", "timestamp"]) || 0));
    $("nav-alert-count").textContent = String(alerts.filter(a => !/recovery|resolved/i.test(String(a.level || a.status || ""))).length);
    $("alert-section-count").textContent = `${alerts.length} EVENT${alerts.length === 1 ? "" : "S"}`;
    const root = $("alert-list");
    if (!alerts.length) { root.innerHTML = '<div class="empty-inline">No active alerts. New camera events will appear here.</div>'; return; }
    root.replaceChildren(...alerts.slice(0, 4).map((a) => {
      const row = document.createElement("div"); row.className = "alert-row";
      const level = String(a.level || a.status || a.verification_state || "preliminary").toLowerCase();
      const kind = String(a.type || a.predicted_class || a.event_type || "traffic event").replaceAll("_", " ");
      const id = esc(a.camera_id || a.camera || "Camera");
      const loc = a.location?.label || a.location_label || "Location unavailable";
      const confirmed = /confirm|model_confirmed/.test(level);
      const recovery = /recover|resolved/.test(level);
      const icon = recovery ? "✓" : "⚠";
      const title = first(a, ["message", "title"]) || `${displayClass(kind)} ${recovery ? "recovery" : "detected"}`;
      const time = first(a, ["captured_at", "created_at", "timestamp"]);
      row.innerHTML = `<span class="alert-symbol ${confirmed ? "confirmed" : ""} ${recovery ? "recovery" : ""}">${icon}</span><span class="alert-main"><span class="alert-title">${escapeHtml(title)}</span><span class="alert-sub">${escapeHtml(id)} · ${escapeHtml(loc)} · ${escapeHtml(confirmed ? "Model-confirmed" : recovery ? "Recovery" : "Preliminary")}</span></span><span class="alert-time">${escapeHtml(fmtTime(time))}</span>`;
      return row;
    }));
  }
  function renderMetrics() {
    const m = state.metrics || {};
    const inference = num(first(m, ["local_inference_count", "inference_count", "local_calls", "ai_calls"]));
    const baseline = num(first(m, ["baseline_inference_count", "baseline_count", "fixed_rate_calls"]));
    let reduction = num(first(m, ["reduction_pct", "measured_reduction_pct", "ai_call_reduction_pct", "calls_saved_pct"]));
    if (reduction === null && inference !== null && baseline > 0) reduction = (1 - inference / baseline) * 100;
    const delay = num(first(m, ["event_detection_delay_sec", "detection_delay_sec", "mean_detection_delay_sec"]));
    const vision = num(first(m, ["openai_vision_calls", "vision_calls", "verification_calls"]));
    const assistant = num(first(m, ["openai_summary_calls", "summary_calls", "assistant_calls"]));
    const semanticBytes = num(first(m, ["semantic_bytes", "semantic_payload_bytes"]));
    const jpegBytes = num(first(m, ["snapshot_bytes", "jpeg_bytes", "preview_bytes"]));
    const bytes = num(first(m, ["total_bytes", "bytes_transmitted"])) ?? (semanticBytes !== null || jpegBytes !== null ? (semanticBytes || 0) + (jpegBytes || 0) : null);
    $("metric-active").textContent = String(state.cameras.filter(isOnline).length);
    $("metric-total").textContent = `/ ${state.cameras.length} total`;
    $("metric-active-note").textContent = state.cameras.length ? `${state.cameras.filter(isOnline).length} currently receiving updates` : "Waiting for camera data";
    const openIncidents = state.alerts.filter(a => !/recovery|resolved/i.test(String(a.level || a.status || ""))).length;
    $("metric-incidents").textContent = state.alerts.length ? String(openIncidents) : "0";
    $("metric-incident-note").textContent = state.alerts.length ? `${state.alerts.length} event${state.alerts.length === 1 ? "" : "s"} in recent history` : "No active incidents";
    $("metric-saved").textContent = fmtPct(reduction);
    $("metric-saved-unit").textContent = reduction === null ? "Awaiting measured baseline" : "fewer calls than fixed 1s";
    $("metric-source").textContent = reduction === null ? "No baseline comparison received" : "Measured by TCC receiver";
    $("metric-inference-count").textContent = inference === null ? "—" : inference.toLocaleString();
    $("metric-baseline-count").textContent = baseline === null ? "—" : baseline.toLocaleString();
    $("metric-delay").textContent = delay === null ? "—" : `${delay.toFixed(1)}s`;
    $("metric-openai-count").textContent = `${vision === null ? "—" : vision} · ${assistant === null ? "—" : assistant}`;
    $("metric-data-bytes").textContent = fmtBytes(bytes);
    $("metrics-updated").textContent = state.lastMetricSuccess ? `Received ${fmtTime(state.lastMetricSuccess)}` : "No metrics received";
  }
  function renderSummary() {
    const s = state.summary;
    const body = $("summary-body");
    if (!s) {
      $("summary-state").textContent = "WAITING";
      $("summary-state").className = "assistant-state";
      body.innerHTML = `<div class="assistant-orb">✳</div><p>${state.healthy ? "Assistant summary is not available yet. Camera state and alerts remain live." : "Situation summaries will appear when the TCC assistant is connected."}</p>`;
      $("summary-time").textContent = state.lastSummarySuccess ? "No summary in response" : "No summary generated";
      return;
    }
    const text = typeof s === "string" ? s : first(s, ["text", "summary", "content", "message"]);
    const fallback = Boolean(s.fallback || s.is_fallback || s.source === "template");
    const status = String(s.status || (fallback ? "fallback" : "ready")).toUpperCase();
    const created = first(s, ["generated_at", "created_at", "updated_at"]);
    const event = first(s, ["event_id", "source_event", "camera_id"]);
    $("summary-state").textContent = fallback ? "TEMPLATE FALLBACK" : status;
    $("summary-state").className = `assistant-state ${!fallback && /ready|complete|generated/i.test(status) ? "ready" : ""}`;
    body.classList.add("has-summary");
    body.innerHTML = `<div class="assistant-orb">✳</div><div><p>${escapeHtml(text || "Summary response did not include readable text.")}</p>${event ? `<span class="summary-event">SOURCE · ${escapeHtml(event)}</span>` : ""}</div>`;
    $("summary-time").textContent = `${fallback ? "Template fallback" : "Generated"} ${fmtTime(created || state.lastSummarySuccess)}`;
  }
  function renderConnection() {
    const ok = state.healthy;
    $("connection-label").textContent = ok ? "TCC connected" : "TCC unavailable";
    $("live-dot").classList.toggle("offline", !ok);
    $("sidebar-link-dot").classList.toggle("ok", ok);
    $("sidebar-link-dot").classList.toggle("bad", !ok && state.lastSuccess !== null);
    $("sidebar-link-label").textContent = ok ? "Connected · live" : state.lastSuccess ? "Connection lost" : "Waiting for receiver";
    $("metric-network").textContent = ok ? "Connected" : "Offline";
    $("metric-network").style.color = ok ? "var(--green)" : "var(--red)";
    $("network-dot").className = `status-dot ${ok ? "green" : ""}`;
    $("network-detail").textContent = ok ? "TCC API responding" : "Check receiver and network";
    $("last-updated").textContent = state.lastSuccess ? fmtTime(state.lastSuccess) : "never";
  }
  function render() {
    renderConnection(); renderCameras(); renderAlerts(); renderMetrics(); renderSummary();
  }
  function updateClock() {
    const now = new Date();
    $("clock").textContent = now.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit", timeZone: "Asia/Bangkok" });
    $("today-label").textContent = now.toLocaleDateString("en-GB", { weekday: "short", day: "2-digit", month: "short", year: "numeric", timeZone: "Asia/Bangkok" });
  }

  const mockData = {
    cameras: [
      { camera_id: "CAM_02", location: { label: "Rangsit demo junction", latitude: 14.068, longitude: 100.607 }, predicted_class: "congestion", confidence: 0.84, score_kind: "local model score", verification_state: "verifying", sampling_interval_sec: 1, priority: "high", captured_at: new Date(Date.now() - 1700).toISOString(), latest_snapshot_url: "", online: true },
      { camera_id: "CAM_01", location: { label: "North corridor · demo", latitude: 14.074, longitude: 100.612 }, predicted_class: "normal_flow", confidence: 0.91, score_kind: "local model score", verification_state: "normal", sampling_interval_sec: 2, priority: "low", captured_at: new Date(Date.now() - 2800).toISOString(), online: true }
    ],
    alerts: [{ camera_id: "CAM_02", location: { label: "Rangsit demo junction" }, type: "congestion", level: "preliminary", message: "Congestion suspected · increased sampling", created_at: new Date(Date.now() - 58000).toISOString() }],
    metrics: { local_inference_count: 168, baseline_inference_count: 224, reduction_pct: 25, event_detection_delay_sec: 2.4, openai_vision_calls: 2, openai_summary_calls: 3, semantic_bytes: 28460, snapshot_bytes: 418000 },
    summary: { text: "Congestion is being monitored at CAM_02 near the Rangsit demo junction. Sampling has increased to every 1 second while the event is verified. Review the latest sampled image before confirming the incident.", source: "mock", status: "ready", generated_at: new Date().toISOString(), event_id: "MOCK-EVENT-01" }
  };

  $("refresh-button").addEventListener("click", () => { poll(); const toast = $("toast"); toast.textContent = "Refreshing TCC data…"; toast.classList.add("show"); setTimeout(() => toast.classList.remove("show"), 1500); });
  updateClock(); render(); poll();
  setInterval(updateClock, 1000);
  setInterval(poll, REFRESH_MS);
})();
