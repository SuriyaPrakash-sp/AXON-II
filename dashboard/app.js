/**
 * app.js — shared state, API layer, and DOM helpers
 * Loaded by all three HTML pages.
 */

const API_BASE = "http://localhost:5000";
const POLL_INTERVAL_MS = 3000;

// ── State ──────────────────────────────────────────────────────────────
const state = {
  predictions: {},          // { N1: "GREEN", ... }
  history: [],              // array of prediction snapshots [{ts, preds}]
  MAX_HISTORY: 20,
  pollTimer: null,
  isLive: false,
  selectedNode: null,
};

// ── Node list ──────────────────────────────────────────────────────────
const NODES = Array.from({ length: 15 }, (_, i) => `N${i + 1}`);
const COLOR_LABELS = { GREEN: "SAFE", YELLOW: "WARNING", RED: "FLOOD" };

// ── API calls ──────────────────────────────────────────────────────────

async function fetchPredictions() {
  try {
    const res = await fetch(`${API_BASE}/predict`);
    if (!res.ok) {
      if (res.status === 503) return null;   // no data yet — not an error
      throw new Error(`HTTP ${res.status}`);
    }
    const data = await res.json();
    return data.predictions;
  } catch (err) {
    console.warn("Prediction fetch failed:", err.message);
    return null;
  }
}

async function postData(snapshot) {
  try {
    const res = await fetch(`${API_BASE}/data`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(snapshot),
    });
    return res.ok;
  } catch {
    return false;
  }
}

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

// ── Mock data generator (fallback when no backend) ──────────────────────
function generateMockSnapshot() {
  const snapshot = {};
  NODES.forEach(node => {
    snapshot[node] = {
      rainfall:      parseFloat((Math.random() * 50).toFixed(2)),
      humidity:      parseFloat((0.4 + Math.random() * 0.5).toFixed(2)),
      cloud_density: parseFloat((Math.random()).toFixed(2)),
      water_level:   parseFloat((0.5 + Math.random() * 3).toFixed(2)),
      rate_of_rise:  parseFloat((Math.random() * 0.2).toFixed(3)),
    };
  });
  return snapshot;
}

function generateMockPredictions() {
  const colors = ["GREEN", "GREEN", "GREEN", "YELLOW", "RED"];
  const result = {};
  NODES.forEach(node => {
    result[node] = colors[Math.floor(Math.random() * colors.length)];
  });
  return result;
}

// ── Color helpers ───────────────────────────────────────────────────────
function colorClass(color) {
  return color || "UNKNOWN";
}

function badgeHtml(color) {
  const cls = { GREEN: "badge-green", YELLOW: "badge-yellow", RED: "badge-red" }[color] || "badge-blue";
  return `<span class="badge ${cls}">${COLOR_LABELS[color] || color}</span>`;
}

function statusCounts(preds) {
  const counts = { GREEN: 0, YELLOW: 0, RED: 0 };
  Object.values(preds).forEach(c => { if (counts[c] !== undefined) counts[c]++; });
  return counts;
}

// ── History management ──────────────────────────────────────────────────
function pushHistory(preds) {
  state.history.push({ ts: new Date(), preds: { ...preds } });
  if (state.history.length > state.MAX_HISTORY) {
    state.history.shift();
  }
}

// ── Polling ─────────────────────────────────────────────────────────────
function startPolling(onUpdate) {
  if (state.pollTimer) return;

  async function tick() {
    let preds = await fetchPredictions();

    // Fallback to mock data if backend is unreachable
    if (!preds) preds = generateMockPredictions();

    state.predictions = preds;
    pushHistory(preds);
    state.isLive = true;
    if (onUpdate) onUpdate(preds);
  }

  tick();
  state.pollTimer = setInterval(tick, POLL_INTERVAL_MS);
}

function stopPolling() {
  clearInterval(state.pollTimer);
  state.pollTimer = null;
  state.isLive = false;
}

// ── Nav highlight ────────────────────────────────────────────────────────
function highlightActiveNav() {
  const path = window.location.pathname.split("/").pop() || "index.html";
  document.querySelectorAll(".nav-link").forEach(el => {
    el.classList.toggle("active", el.dataset.page === path);
  });
}

// ── Update nav status dot ────────────────────────────────────────────────
function setNavStatus(live) {
  const dot  = document.getElementById("status-dot");
  const text = document.getElementById("status-text");
  if (!dot || !text) return;
  if (live) {
    dot.className  = "status-dot live";
    text.textContent = "Live";
  } else {
    dot.className  = "status-dot error";
    text.textContent = "Offline";
  }
}

// ── Format helpers ───────────────────────────────────────────────────────
function formatTime(date) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

// ── Init on load ─────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  highlightActiveNav();

  // Send mock data to backend once so /predict has data
  postData(generateMockSnapshot()).then(ok => {
    setNavStatus(ok);
  });
});