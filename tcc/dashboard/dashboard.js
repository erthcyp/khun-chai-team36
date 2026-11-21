(() => {
  "use strict";

  const STALE_AFTER_MS = 12000;
  const CAMERA_IDS = Array.from({ length: 6 }, (_, index) => `CAM_0${index + 1}`);
  const params = new URLSearchParams(location.search);
  const requestedInterval = Number(params.get("pollMs") || 1000);
  const REFRESH_MS = Number.isFinite(requestedInterval)
    ? Math.max(500, Math.min(60000, requestedInterval)) : 1000;
  const mockMode = params.get("mock") === "1";
  const apiBase = (params.get("api") || "").replace(/\/$/, "");
  const state = { cameras: [], alerts: [], metrics: null, connected: false, updatedAt: null, selectedCameraId: null };
  const $ = (id) => document.getElementById(id);
  let summaryLanguage = "th";
  try { if (localStorage.getItem("roadsense-summary-language") === "en") summaryLanguage = "en"; } catch (_) {}
  const classNames = {
    normal_flow: "จราจรปกติ",
    congestion: "จราจรติดขัด",
    accident: "อุบัติเหตุ",
    invalid_scene: "กล้องขัดข้อง"
  };
  const stateNames = {
    normal: "เฝ้าระวัง",
    verifying: "กำลังตรวจสอบ",
    alert: "แจ้งเตือน",
    recovery: "กำลังกลับสู่ปกติ",
    invalid: "สัญญาณกล้องผิดปกติ",
    offline: "ออฟไลน์"
  };

  function number(value) {
    if (value === null || value === undefined || value === "") return null;
    const result = Number(value);
    return Number.isFinite(result) ? result : null;
  }
  function first(obj, keys) {
    for (const key of keys) if (obj && obj[key] !== null && obj[key] !== undefined) return obj[key];
    return null;
  }
  function records(payload, keys) {
    if (Array.isArray(payload)) return payload;
    for (const key of keys) if (Array.isArray(payload?.[key])) return payload[key];
    return [];
  }
  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
  }
  function normalizeClass(value) {
    return String(value || "").toLowerCase().replace(/[ -]/g, "_");
  }
  function thaiClass(value) {
    return classNames[normalizeClass(value)] || "กำลังรอผลวิเคราะห์";
  }
  function thaiTime(value) {
    if (!value) return "—";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? "—" : date.toLocaleTimeString("th-TH", { hour: "2-digit", minute: "2-digit", second: "2-digit", timeZone: "Asia/Bangkok" });
  }
  function ageLabel(value) {
    if (!value) return "ยังไม่มีเฟรม";
    const timestamp = new Date(value).getTime();
    if (!Number.isFinite(timestamp)) return "ไม่ทราบเวลา";
    const seconds = Math.max(0, Math.floor((Date.now() - timestamp) / 1000));
    if (seconds < 2) return "ล่าสุด";
    if (seconds < 60) return `${seconds} วินาทีที่แล้ว`;
    return `${Math.floor(seconds / 60)} นาทีที่แล้ว`;
  }
  function capturedAt(camera) {
    return first(camera, ["captured_at", "updated_at", "received_at", "last_seen_at", "timestamp"]);
  }
  function isOnline(camera) {
    const explicit = first(camera, ["online", "is_online", "connected"]);
    if (explicit !== null) return Boolean(explicit);
    const at = capturedAt(camera);
    return at ? Date.now() - new Date(at).getTime() <= STALE_AFTER_MS : false;
  }
  function cameraLocation(camera) {
    if (camera.placeholder) return "รอตำแหน่งกล้อง";
    const value = camera.location || {};
    return String(first(value, ["label", "name"]) || camera.location_label || "ไม่ระบุตำแหน่ง");
  }
  function prediction(camera) {
    return first(camera, ["predicted_class", "prediction", "final_class", "classification"]);
  }
  function naturalLanguage(camera) {
    const summary = camera?.meaning?.[`summary_${summaryLanguage}`] ?? camera?.[`summary_${summaryLanguage}`];
    if (typeof summary === "string") return summary.trim();
    if (summaryLanguage === "en") return "";
    const value = first(camera, [
      "natural_language_description", "natural_language", "description_th",
      "explanation_th", "human_readable_text", "description", "explanation", "summary"
    ]);
    if (typeof value === "string") return value.trim();
    if (value && typeof value === "object") return String(first(value, ["th", "text", "content"]) || "").trim();
    return "";
  }
  function cameraState(camera) {
    return String(camera.verification_state || camera.state || "normal").toLowerCase();
  }
  function openAlert(alert) {
    return !/recovery|resolved|closed/i.test(String(alert.level || alert.status || ""));
  }
  function cameraIndex(camera) {
    const id = String(camera.camera_id || camera.id || "").trim();
    const match = id.match(/^CAM[_ -]?0?([1-6])$/i);
    return match ? Number(match[1]) : null;
  }
  function normalizeCameras(cameras) {
    const byIndex = new Map();
    for (const camera of cameras) {
      const index = cameraIndex(camera);
      if (index === null) continue;
      const current = byIndex.get(index);
      const currentTime = new Date(capturedAt(current) || 0).getTime();
      const nextTime = new Date(capturedAt(camera) || 0).getTime();
      if (!current || nextTime >= currentTime) {
        byIndex.set(index, { ...camera, camera_id: `CAM_0${index}` });
      }
    }
    return [...byIndex.entries()].sort(([a], [b]) => a - b).map(([, camera]) => camera);
  }
  function displayCameras(cameras) {
    const byId = new Map(cameras.map((camera) => [camera.camera_id, camera]));
    return CAMERA_IDS.map((cameraId) => byId.get(cameraId) || { camera_id: cameraId, placeholder: true });
  }
  async function json(path) {
    const response = await fetch(`${apiBase}${path}`, { cache: "no-store", signal: AbortSignal.timeout(5000), headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(`${path}: ${response.status}`);
    return response.json();
  }

  function setTheme(theme) {
    const next = theme === "light" ? "light" : "dark";
    document.body.dataset.theme = next;
    try { localStorage.setItem("roadsense-theme", next); } catch (_) {}
    $("theme-icon").textContent = next === "dark" ? "☀" : "☾";
    $("theme-label").textContent = next === "dark" ? "โหมดสว่าง" : "โหมดมืด";
    $("theme-toggle").setAttribute("aria-label", next === "dark" ? "เปลี่ยนเป็นโหมดสว่าง" : "เปลี่ยนเป็นโหมดมืด");
    document.querySelector('meta[name="theme-color"]').content = next === "dark" ? "#0d141e" : "#e9eef2";
  }
  function initTheme() {
    let saved = "dark";
    try { saved = localStorage.getItem("roadsense-theme") || "dark"; } catch (_) {}
    setTheme(saved);
    $("theme-toggle").addEventListener("click", () => setTheme(document.body.dataset.theme === "dark" ? "light" : "dark"));
  }

  function makeCamera(camera, variant = "", previousImage = null) {
    const cameraId = String(camera.camera_id || camera.id || "กล้องไม่ทราบรหัส");
    const waiting = Boolean(camera.placeholder);
    const status = cameraState(camera);
    const online = isOnline(camera);
    const picture = camera.snapshot;
    const latestClass = normalizeClass(prediction(camera));
    const className = picture ? normalizeClass(prediction(picture)) : latestClass;
    const isAlert = Boolean(camera.alert);
    const isInvalid = status === "invalid";
    const isVerify = status === "verifying" || status === "pending";
    const trafficKind = waiting ? "waiting" : className;
    const trafficLabel = waiting ? "รอข้อมูล" : thaiClass(trafficKind);
    const interval = number(camera.sampling_interval_sec);
    const timestamp = capturedAt(camera);
    const imageTimestamp = picture?.captured_at || timestamp;
    const snapshot = first(camera, ["latest_snapshot_url", "snapshot_url", "preview_url", "image_url"]);
    const wrapper = document.createElement("article");
    wrapper.dataset.cameraId = cameraId;
    wrapper.className = `camera-card ${variant} ${waiting ? "is-placeholder" : ""}`.trim();

    const head = document.createElement("div");
    head.className = "camera-card-head";
    const title = document.createElement("div");
    title.className = "camera-title";
    const id = document.createElement("div");
    id.className = "camera-id";
    id.textContent = cameraId.replace("_", " ");
    const loc = document.createElement("div");
    loc.className = "camera-location";
    loc.textContent = cameraLocation(camera);
    title.append(id, loc);
    const badge = document.createElement("span");
    badge.className = `traffic-badge ${trafficKind}`;
    badge.innerHTML = `<span class="badge-dot"></span>${escapeHtml(trafficLabel)}`;
    head.append(title, badge);

    const preview = document.createElement("div");
    preview.className = "camera-preview";
    if (snapshot) {
      const url = new URL(String(snapshot), apiBase || location.origin);
      if (!picture && camera.frame_id) url.searchParams.set("frame", camera.frame_id);
      const image = previousImage?.src === url.href ? previousImage : document.createElement("img");
      if (image.src !== url.href) image.src = url.href;
      image.alt = `ภาพเฟรมล่าสุดจาก ${cameraId}`;
      image.loading = "lazy";
      image.onerror = () => {
        image.remove();
        preview.insertAdjacentHTML("afterbegin", '<div class="preview-placeholder"><span class="road-shape"></span></div>');
      };
      preview.appendChild(image);
    } else {
      preview.innerHTML = '<div class="preview-placeholder"><span class="road-shape"></span></div>';
    }
    const chip = document.createElement("span");
    chip.className = "snapshot-chip";
    chip.textContent = snapshot ? "เฟรมล่าสุด" : "รอภาพจากกล้อง";
    const age = document.createElement("span");
    age.className = "snapshot-age";
    age.textContent = ageLabel(imageTimestamp);
    preview.append(chip, age);
    if (variant === "wall-camera-card") {
      const open = document.createElement("button");
      open.type = "button";
      open.className = "camera-preview-open";
      open.dataset.openCamera = cameraId;
      open.setAttribute("aria-label", `ดูรายละเอียด ${cameraId}`);
      preview.append(open);
    }

    const intervalLabel = document.createElement("div");
    intervalLabel.className = "camera-interval";
    intervalLabel.textContent = interval === null ? "รอข้อมูล" : `${interval} วินาที`;
    const foot = document.createElement("div");
    foot.className = "camera-card-foot";
    const stateLabel = document.createElement("span");
    stateLabel.className = `camera-state-label ${waiting ? "waiting" : isVerify ? "verify" : isAlert || isInvalid ? "alert" : ""}`;
    stateLabel.innerHTML = `<span class="badge-dot"></span>${escapeHtml(waiting ? "รอข้อมูลจาก RSU" : !online ? stateNames.offline : stateNames[status] || stateNames.normal)}`;
    const updated = document.createElement("span");
    updated.textContent = timestamp ? `อัปเดต ${thaiTime(timestamp)}` : "ยังไม่ได้รับข้อมูล";
    foot.append(stateLabel, updated);
    const timing = document.createElement("div");
    timing.className = "camera-timing";
    const capture = picture || camera;
    const inference = number(capture.inference_ms);
    const playback = number(capture.video_timestamp_sec);
    const captureToTcc = Date.parse(capture.received_at) - Date.parse(capture.captured_at);
    const alertLevel = camera.alert?.level === "confirmed" ? "ยืนยันแล้ว"
      : camera.alert?.level === "preliminary" ? "เบื้องต้น" : null;
    const displayedFrame = picture?.frame_id || camera.frame_id;
    timing.textContent = [
      displayedFrame && `${displayedFrame} · เวลาเก็บเฟรม ${thaiTime(capture.captured_at)}`,
      playback !== null && `เวลาบนคลิป ${playback.toFixed(2)} s`,
      inference !== null && `เวลารันโมเดล ${inference.toFixed(0)} ms`,
      Number.isFinite(captureToTcc) && captureToTcc >= 0 && `เก็บเฟรมถึง TCC ${captureToTcc} ms`,
      picture?.image_received_at && `TCC รับภาพ ${thaiTime(picture.image_received_at)}`,
      !waiting && !online && `สถานะล่าสุดก่อนออฟไลน์: ${stateNames[status] || status}`,
      alertLevel && `แจ้งเตือนล่าสุด ${alertLevel}`,
    ].filter(Boolean).join(" · ");
    const latest = document.createElement("div");
    latest.className = "camera-timing";
    const latestLabel = summaryLanguage === "en"
      ? ({normal_flow: "Normal flow", congestion: "Congestion", accident: "Accident",
          invalid_scene: "Invalid scene"}[latestClass] || latestClass)
      : thaiClass(latestClass);
    latest.lang = summaryLanguage;
    if (!waiting) {
      const label = summaryLanguage === "en" ? "Latest prediction" : "ผลทำนายล่าสุด";
      const pending = summaryLanguage === "en" ? "waiting for image" : "รอภาพของเฟรมนี้";
      latest.textContent = `${label}: ${latestLabel} · ${camera.frame_id || "—"}${camera.snapshot_pending ? ` · ${pending}` : ""}`;
    }
    const coords = camera.location || {};
    const lat = number(coords.latitude), lon = number(coords.longitude);
    const coordinateLine = document.createElement("div");
    coordinateLine.className = "location-coordinates";
    coordinateLine.textContent = lat !== null && lon !== null ? `${lat.toFixed(4)}, ${lon.toFixed(4)}` : "";
    const explanation = naturalLanguage(camera);
    const description = document.createElement("section");
    description.className = `ai-description-panel${explanation ? "" : " is-waiting"}`;
    description.setAttribute("aria-label", `คำอธิบายเหตุการณ์ของ ${cameraId}`);
    const descriptionHeading = document.createElement("div");
    descriptionHeading.className = "ai-description-heading";
    const heading = document.createElement("h3");
    heading.textContent = "คำอธิบายเหตุการณ์";
    const source = document.createElement("span");
    source.className = "description-source";
    const meaningSource = camera.meaning?.source || camera.source;
    source.textContent = mockMode ? "ตัวอย่างจำลอง" : camera.meaning?.fallback_reason === "image_timeout"
      ? (summaryLanguage === "th" ? "ไม่ได้รับภาพประกอบ" : "Snapshot unavailable")
      : meaningSource === "fallback" ? "ข้อความสำรอง"
      : meaningSource === "json+image" ? "AI + ภาพ" : explanation ? "จาก TCC" : "รอข้อมูล";
    descriptionHeading.append(heading, source);
    const descriptionText = document.createElement("p");
    descriptionText.lang = summaryLanguage;
    descriptionText.textContent = explanation || (summaryLanguage === "th"
      ? (camera.meaning_pending ? "กำลังรอสรุปจาก TCC" : "ยังไม่มีคำอธิบายสำหรับกล้องนี้")
      : (camera.meaning_pending ? "Waiting for the TCC summary" : "No English summary yet"));
    description.append(descriptionHeading, descriptionText);
    const meaning = camera.meaning;
    if (meaning?.source_frame_id || meaning?.generated_at) {
      const provenance = document.createElement("small");
      provenance.className = "camera-timing";
      provenance.textContent = `สรุปจาก ${meaning.source_frame_id || "—"} · เก็บเฟรม ${thaiTime(meaning.source_captured_at)} · สร้างสรุป ${thaiTime(meaning.generated_at)}`;
      description.append(provenance);
    }
    wrapper.append(head, preview, intervalLabel, foot, coordinateLine, timing, latest, description);
    return wrapper;
  }

  function renderCameraWall(cameras, receivedCount) {
    const root = $("wall-grid");
    const previousImages = new Map([...root.children].map((card) => [
      card.dataset.cameraId, card.querySelector(".camera-preview img"),
    ]));
    root.replaceChildren(...cameras.map((camera) =>
      makeCamera(camera, "wall-camera-card", previousImages.get(camera.camera_id))
    ));
    $("wall-count").textContent = `${receivedCount} จาก 6 กล้องที่มีข้อมูล`;
  }

  function renderCameras() {
    const received = normalizeCameras(state.cameras);
    const cameras = displayCameras(received);
    const activeCount = received.filter(isOnline).length;
    $("online-count").textContent = `${activeCount} กล้องออนไลน์`;
    const hasAlert = state.alerts.some(openAlert);
    $("overall-state").textContent = hasAlert ? "มีเหตุการณ์ที่ต้องตรวจสอบ" : received.length ? "เฝ้าระวังตามปกติ" : "รอข้อมูลจาก RSU";
    $("metric-context").textContent = received.length
      ? `${mockMode ? "รอบจำลอง" : "ข้อมูลจริง"} · ${received.length} กล้อง`
      : `${mockMode ? "รอบจำลอง" : "ข้อมูลจริง"} · รอข้อมูล`;
    const focusedCamera = document.activeElement?.dataset.openCamera;
    renderCameraWall(cameras, received.length);
    renderCameraDetail(cameras);
    if (focusedCamera) {
      $("wall-grid").querySelector(`[data-open-camera="${CSS.escape(focusedCamera)}"]`)?.focus({ preventScroll: true });
    }
  }

  function renderCameraDetail(cameras) {
    const dialog = $("camera-detail");
    if (!dialog.open) return;
    const camera = cameras.find((item) => item.camera_id === state.selectedCameraId);
    if (!camera) return;
    $("camera-detail-title").textContent = camera.camera_id.replace("_", " ");
    const content = $("camera-detail-content");
    const image = content.querySelector(".camera-preview img");
    content.replaceChildren(makeCamera(camera, "detail-camera-card", image));
  }

  function renderAlerts() {
    const alerts = [...state.alerts].sort((a, b) => {
      const ta = new Date(first(a, ["captured_at", "created_at", "timestamp"]) || 0).getTime();
      const tb = new Date(first(b, ["captured_at", "created_at", "timestamp"]) || 0).getTime();
      return tb - ta;
    });
    const activeCount = alerts.filter(openAlert).length;
    $("alert-count").textContent = `${activeCount} รายการที่เปิดอยู่`;
    const root = $("alert-list");
    if (!alerts.length) {
      root.innerHTML = '<p class="empty-state">ไม่มีรายการแจ้งเตือน</p>';
      return;
    }
    root.replaceChildren(...alerts.slice(0, 8).map((alert) => {
      const row = document.createElement("div");
      row.className = "alert-item";
      const level = String(alert.level || alert.status || "preliminary").toLowerCase();
      const recovery = /recovery|resolved/.test(level);
      const confirmed = /confirm/.test(level);
      const type = normalizeClass(alert.type || alert.predicted_class || alert.event_type);
      const cameraId = alert.camera_id || alert.camera || "กล้อง";
      const place = alert.location?.label || alert.location_label || "ไม่ระบุตำแหน่ง";
      const mark = document.createElement("span");
      mark.className = `alert-mark ${recovery ? "recovery" : confirmed ? "confirmed" : type}`;
      mark.textContent = recovery ? "✓" : type === "accident" ? "!" : "•";
      const copy = document.createElement("div");
      copy.className = "alert-copy";
      const title = document.createElement("div");
      title.className = "alert-title";
      title.textContent = alert.message || `${thaiClass(type)}${recovery ? " · กำลังกลับสู่ปกติ" : confirmed ? " · ยืนยันจากโมเดล" : " · แจ้งเตือนเบื้องต้น"}`;
      const detail = document.createElement("div");
      detail.className = "alert-detail";
      detail.textContent = `${cameraId} · ${place}`;
      const time = document.createElement("div");
      time.className = "alert-time";
      time.textContent = thaiTime(first(alert, ["captured_at", "created_at", "timestamp"]));
      copy.append(title, detail, time);
      row.append(mark, copy);
      return row;
    }));
  }

  function renderMetrics() {
    const metrics = state.metrics || {};
    const used = number(first(metrics, ["local_inference_count", "inference_count", "local_calls", "ai_calls"]));
    const baseline = number(first(metrics, ["baseline_inference_count", "baseline_count", "fixed_rate_calls"]));
    let reduction = number(first(metrics, ["reduction_pct", "measured_reduction_pct", "ai_call_reduction_pct", "calls_saved_pct"]));
    if (reduction === null && used !== null && baseline > 0) reduction = (1 - used / baseline) * 100;
    $("adaptive-count").textContent = used === null ? "—" : used.toLocaleString("th-TH");
    $("baseline-count").textContent = baseline === null ? "—" : baseline.toLocaleString("th-TH");
    $("reduction-percent").textContent = reduction === null ? "—" : `${Math.max(0, reduction).toFixed(reduction % 1 ? 1 : 0)}%`;
    const scale = Math.max(used || 0, baseline || 0);
    $("adaptive-bar").style.width = scale > 0 && used !== null ? `${Math.max(3, used / scale * 100)}%` : "0%";
    $("baseline-bar").style.width = scale > 0 && baseline !== null ? `${Math.max(3, baseline / scale * 100)}%` : "0%";
  }

  async function poll() {
    if (mockMode) {
      state.connected = true;
      state.cameras = normalizeCameras([
        { camera_id: "CAM_01", location: { label: "ถนนสายหลัก", latitude: 13.7563, longitude: 100.5018 }, predicted_class: "normal_flow", natural_language_description: "ตัวอย่างจำลอง: จากภาพล่าสุด รถบนถนนสายหลักเคลื่อนตัวต่อเนื่อง ยังไม่พบสัญญาณการจราจรติดขัด", verification_state: "normal", sampling_interval_sec: 2, priority: "low", captured_at: new Date(Date.now() - 1400).toISOString(), online: true },
        { camera_id: "CAM_02", location: { label: "ทางขึ้นสะพาน", latitude: 13.7591, longitude: 100.5056 }, predicted_class: "normal_flow", natural_language_description: "ตัวอย่างจำลอง: รถบนทางขึ้นสะพานเคลื่อนตัวได้ตามปกติ ไม่พบเหตุผิดปกติในภาพล่าสุด", verification_state: "normal", sampling_interval_sec: 2, priority: "low", captured_at: new Date(Date.now() - 2100).toISOString(), online: true },
        { camera_id: "CAM_03", location: { label: "พื้นที่จำลอง CAM_03" }, predicted_class: "normal_flow", natural_language_description: "ตัวอย่างจำลอง: ภาพจาก CAM 03 อยู่ในสถานะปกติ ยังไม่พบเหตุที่ต้องแจ้งเตือน", verification_state: "normal", sampling_interval_sec: 2, priority: "low", captured_at: new Date(Date.now() - 2700).toISOString(), online: true },
        { camera_id: "CAM_04", location: { label: "พื้นที่จำลอง CAM_04" }, predicted_class: "normal_flow", natural_language_description: "ตัวอย่างจำลอง: ภาพจาก CAM 04 อยู่ในสถานะปกติ ยังไม่พบเหตุที่ต้องแจ้งเตือน", verification_state: "normal", sampling_interval_sec: 2, priority: "low", captured_at: new Date(Date.now() - 3200).toISOString(), online: true },
        { camera_id: "CAM_05", location: { label: "พื้นที่จำลอง CAM_05" }, predicted_class: "normal_flow", natural_language_description: "ตัวอย่างจำลอง: ภาพจาก CAM 05 อยู่ในสถานะปกติ ยังไม่พบเหตุที่ต้องแจ้งเตือน", verification_state: "normal", sampling_interval_sec: 2, priority: "low", captured_at: new Date(Date.now() - 3800).toISOString(), online: true },
        { camera_id: "CAM_06", location: { label: "พื้นที่จำลอง CAM_06" }, predicted_class: "normal_flow", natural_language_description: "ตัวอย่างจำลอง: ภาพจาก CAM 06 อยู่ในสถานะปกติ ยังไม่พบเหตุที่ต้องแจ้งเตือน", verification_state: "normal", sampling_interval_sec: 2, priority: "low", captured_at: new Date(Date.now() - 4300).toISOString(), online: true }
      ]);
      state.alerts = [];
      state.metrics = { local_inference_count: 40, baseline_inference_count: 80, reduction_pct: 50 };
      state.updatedAt = new Date();
      render();
      return;
    }
    const [health, cameras, alerts, metrics] = await Promise.allSettled([
      json("/health"), json("/api/v1/cameras"), json("/api/v1/alerts"), json("/api/v1/metrics")
    ]);
    const healthOk = health.status === "fulfilled";
    const camerasOk = cameras.status === "fulfilled";
    const alertsOk = alerts.status === "fulfilled";
    const metricsOk = metrics.status === "fulfilled";
    if (camerasOk) state.cameras = normalizeCameras(records(cameras.value, ["cameras", "items", "data"]));
    if (alertsOk) state.alerts = records(alerts.value, ["alerts", "events", "items", "data"]);
    if (metricsOk) state.metrics = metrics.value?.metrics || metrics.value;
    state.connected = healthOk && camerasOk && state.cameras.some(isOnline);
    if (state.connected) state.updatedAt = new Date();
    render();
  }

  function render() {
    const connected = state.connected;
    $("connection-dot").classList.toggle("offline", !connected);
    document.querySelector(".small-live-dot").classList.toggle("offline", !connected);
    $("connection-text").textContent = connected ? "RSU เชื่อมต่อ" : "ไม่พบการเชื่อมต่อ RSU";
    $("data-mode").textContent = mockMode ? "ข้อมูลจำลอง" : "ข้อมูลจริง";
    $("updated-at").textContent = state.updatedAt ? `อัปเดต ${thaiTime(state.updatedAt)}` : "ยังไม่ได้รับข้อมูล";
    renderCameras();
    renderAlerts();
    renderMetrics();
  }

  initTheme();
  const languageToggle = $("summary-language-toggle");
  function updateLanguageToggle() {
    const english = summaryLanguage === "en";
    languageToggle.textContent = english ? "ไทย" : "English";
    languageToggle.lang = english ? "th" : "en";
    languageToggle.setAttribute("aria-label", english
      ? "Switch event descriptions to Thai"
      : "แสดงคำอธิบายเหตุการณ์เป็นภาษาอังกฤษ");
  }
  updateLanguageToggle();
  languageToggle.addEventListener("click", () => {
    summaryLanguage = summaryLanguage === "th" ? "en" : "th";
    try { localStorage.setItem("roadsense-summary-language", summaryLanguage); } catch (_) {}
    updateLanguageToggle();
    renderCameras();
  });
  $("wall-grid").addEventListener("click", (event) => {
    const button = event.target.closest("[data-open-camera]");
    if (!button) return;
    state.selectedCameraId = button.dataset.openCamera;
    $("camera-detail").showModal();
    renderCameraDetail(displayCameras(normalizeCameras(state.cameras)));
  });
  $("camera-detail").addEventListener("close", () => {
    const id = state.selectedCameraId;
    state.selectedCameraId = null;
    if (id) $("wall-grid").querySelector(`[data-open-camera="${CSS.escape(id)}"]`)?.focus();
  });
  if (mockMode) $("mock-ribbon").hidden = false;
  render();
  async function pollNext() {
    try {
      await poll();
    } finally {
      if (!mockMode) setTimeout(pollNext, REFRESH_MS);
    }
  }
  pollNext();
  setInterval(() => { $("clock").textContent = new Date().toLocaleTimeString("th-TH", { hour: "2-digit", minute: "2-digit", second: "2-digit", timeZone: "Asia/Bangkok" }); }, 1000);
})();
