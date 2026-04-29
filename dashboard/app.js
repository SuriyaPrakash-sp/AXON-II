/**
 * app.js — shared state, API layer, and DOM helpers
 * Loaded by all three HTML pages.
 */

const API_BASE = "http://localhost:5000";
const POLL_INTERVAL_MS = 3000;

// ── State ──────────────────────────────────────────────────────────────
const state = {
  predictions: {},          // { N1: "GREEN", ... }
  latestSensorData: {},     // { N1: { rainfall, humidity, ... }, ... }
  prevSensorData: {},       // previous snapshot for delta display
  latestSosData: null,      // { node_id, mapped_node, lat, lon, battery, active }
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

async function fetchLatestSensorData() {
  try {
    const res = await fetch(`${API_BASE}/latest`);
    if (!res.ok) {
      if (res.status === 503) return null; // no data yet — not an error
      throw new Error(`HTTP ${res.status}`);
    }

    const data = await res.json();

    // Accept either { nodes: {...} } or direct { N1: {...} } payloads.
    if (data && typeof data === "object") {
      if (data.nodes && typeof data.nodes === "object") return data.nodes;
      return data;
    }

    return null;
  } catch (err) {
    console.warn("Latest sensor fetch failed:", err.message);
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

async function fetchLatestSosData() {
  try {
    const res = await fetch(`${API_BASE}/sos/latest`);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    return await res.json();
  } catch (err) {
    console.warn("Latest SOS fetch failed:", err.message);
    return null;
  }
}

function isPlaceholderSnapshot(snapshot) {
  if (!snapshot || typeof snapshot !== "object") return true;

  const nodes = Object.values(snapshot);
  if (nodes.length === 0) return true;

  return nodes.every(node =>
    node
    && Number(node.rainfall ?? 0) === 0
    && Number(node.water_level ?? 0) === 0
    && Number(node.rate_of_rise ?? 0) === 0
    && Number(node.humidity ?? 0) === 50
    && Number(node.cloud_density ?? 0) === 50
  );
}

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

// ── Realistic mock data generator with smooth gradual spatial-temporal transitions ───────

// --- Node graph structure (adjacency map) ---
const NODE_EDGES = [
  ["N1", "N3"], ["N2", "N3"],
  ["N3", "N7"],
  ["N4", "N6"], ["N5", "N6"],
  ["N6", "N8"],
  ["N10", "N11"], ["N11", "N13"], ["N12", "N13"],
  ["N7", "N9"], ["N8", "N9"],
  ["N13", "N14"], ["N9", "N14"],
  ["N14", "N15"]
];
const NODE_NEIGHBORS = (() => {
  const map = {};
  NODES.forEach(n => map[n] = []);
  NODE_EDGES.forEach(([a, b]) => {
    map[a].push(b);
    map[b].push(a);
  });
  return map;
})();

// --- Internal state for mock transitions with staged color transitions ---
const mockState = {
  stagedEvents: [], // [{node, state, tick, stage, fadeOut}], state: 'waiting'|'growing'|'peak'|'fading'
  colorMap: {},
  tickCount: 0,
  timeToNextEvent: 0,
};

function nodesWithinSteps(start, steps) {
  const res = new Set([start]);
  let frontier = [start];
  for (let i = 0; i < steps; i++) {
    const next = [];
    frontier.forEach(n => {
      NODE_NEIGHBORS[n].forEach(nb => {
        if (!res.has(nb)) {
          res.add(nb);
          next.push(nb);
        }
      });
    });
    frontier = next;
  }
  return Array.from(res);
}

// Helper: Pick a node not already a center of an active staged event
function pickFloodSeedNode() {
  // Exclude nodes already involved in any stage of an event
  const coveredNodes = new Set();
  mockState.stagedEvents.forEach(e => {
    if (e.state !== "done" && e.node) {
      coveredNodes.add(e.node); // Avoid overlap of core
      nodesWithinSteps(e.node, 2).forEach(n => coveredNodes.add(n)); // (optionally can loosen this)
    }
  });
  // Prefer core nodes (not edge nodes)
  const coreNodes = NODES.filter(n =>
    NODE_NEIGHBORS[n].length >= 2 &&
    !coveredNodes.has(n)
  );
  if (!coreNodes.length) return null;
  return coreNodes[Math.floor(Math.random() * coreNodes.length)];
}

/**
 * Stages:
 * - 'waiting': all green, candidate node gets set to 'growing' (its core turns yellow)
 * - 'growing': node center yellow, then after a step, neighbors yellow, then center-> red
 * - 'peak': node center red, neighbors yellow
 * - 'fading': color retreats in reverse: center red -> center yellow, neighbors yellow -> green, then center green
 * Each stage has a defined tick budget for transition, making changes only every N ticks for smoothness.
 */
function updateStagedEvents() {
  // Age and update staged events
  const updated = [];
  for (let event of mockState.stagedEvents) {
    if (!event) continue;
    let { node, state, startedTick, stageStep, fadeOut } = event;
    const age = mockState.tickCount - startedTick;
    let transition = false;

    // Transition logic
    if (state === "growing") {
      if (stageStep === 0) {
        // Stage 0: turn center node yellow
        event.stageStep++;
        transition = true;
      } else if (stageStep === 1 && age >= 1) {
        // Stage 1: turn neighbors yellow
        event.stageStep++;
        transition = true;
      } else if (stageStep === 2 && age >= 2) {
        // Stage 2: turn center node red
        event.stageStep++;
        event.state = "peak";
        event.peakTick = mockState.tickCount;
        transition = true;
      }
    } else if (state === "peak") {
      // After a while (random 4-7 ticks), begin fading process
      if (!event.peakTick) event.peakTick = mockState.tickCount;
      if (mockState.tickCount - event.peakTick >= (fadeOut ? 1 : (4 + Math.floor(Math.random() * 4)))) {
        event.state = "fading";
        event.fadeStep = 0;
        transition = true;
      }
    } else if (state === "fading") {
      // Fade: red -> yellow (center), yellow (neighbors) -> green, then all green
      if (event.fadeStep === 0) {
        // Center red -> yellow, neighbors yellow remain
        event.fadeStep++;
        transition = true;
      } else if (event.fadeStep === 1) {
        // Center yellow -> green, neighbors yellow -> green
        event.fadeStep++;
        event.state = "done";
        transition = true;
      }
    }
    if (event.state !== "done") updated.push(event);
  }
  mockState.stagedEvents = updated;
}

// Generate new event only if no ongoing event and delay exceeded
function tryInjectStagedEvent() {
  if (
    mockState.stagedEvents.length === 0 &&
    mockState.timeToNextEvent <= 0
  ) {
    const seed = pickFloodSeedNode();
    if (seed) {
      mockState.stagedEvents.push({
        node: seed,
        state: "growing",
        startedTick: mockState.tickCount,
        stageStep: 0,  // 0: center yellow, 1: neighbors, 2: center red
        fadeOut: false,
      });
      // Time to next event: 18-30s depending on random
      mockState.timeToNextEvent = 6 + Math.floor(Math.random() * 5);
    }
  }
}

// Main: Generate mock predictions with staged, smooth transitions
function generateMockPredictions() {
  mockState.tickCount++;
  // Decrement next event window
  if (mockState.timeToNextEvent > 0) mockState.timeToNextEvent--;

  // Progress/fade staged events
  updateStagedEvents();

  // Try to inject new event if possible
  tryInjectStagedEvent();

  // If peak events have aged enough, mark them for fadeout (if not yet fading)
  mockState.stagedEvents.forEach(ev => {
    if (ev.state === "peak" && (mockState.tickCount - ev.peakTick > 4)) {
      ev.fadeOut = true;
    }
  });

  // Final colorMap build (by overlaying all events/stages)
  const colorMap = {};
  NODES.forEach(node => (colorMap[node] = "GREEN"));

  for (const ev of mockState.stagedEvents) {
    if (!ev) continue;
    // Growing stages
    if (ev.state === "growing") {
      if (ev.stageStep === 1) {
        // Stage 1: center yellow
        colorMap[ev.node] = "YELLOW";
      } else if (ev.stageStep === 2) {
        // Stage 2: neighbors yellow
        colorMap[ev.node] = "YELLOW";
        nodesWithinSteps(ev.node, 1).forEach(nb => colorMap[nb] = "YELLOW");
      } else if (ev.stageStep >= 3) {
        // Center red, neighbors yellow
        colorMap[ev.node] = "RED";
        nodesWithinSteps(ev.node, 1).forEach(nb => colorMap[nb] = "YELLOW");
      }
    } else if (ev.state === "peak") {
      colorMap[ev.node] = "RED";
      nodesWithinSteps(ev.node, 1).forEach(nb => colorMap[nb] = "YELLOW");
    } else if (ev.state === "fading") {
      // Fade out: first, center red->yellow; then all back to green
      if (ev.fadeStep === 0) {
        colorMap[ev.node] = "YELLOW";
        nodesWithinSteps(ev.node, 1).forEach(nb => colorMap[nb] = "YELLOW");
      } else if (ev.fadeStep >= 1) {
        // All green
      }
    }
  }
  mockState.colorMap = { ...colorMap };
  return mockState.colorMap;
}

// Generate realistic mock sensor snapshot based on current prediction state
function generateMockSnapshot() {
  const preds = mockState.colorMap && Object.keys(mockState.colorMap).length ? mockState.colorMap : generateMockPredictions();
  const snapshot = {};
  NODES.forEach(node => {
    // Let sensor values reflect status & diffuse spatially
    let baseWL, baseRain;

    // RED (flood): high water/rain
    if (preds[node] === "RED") {
      baseWL = 90 + Math.random() * 10;   // 90-100 cm
      baseRain = 24 + Math.random() * 24; // heavy rain
    }
    // YELLOW (warning): mid-high water, smaller rain
    else if (preds[node] === "YELLOW") {
      baseWL = 60 + Math.random() * 30;   // 60-90 cm
      baseRain = 12 + Math.random() * 20;
    }
    // GREEN (safe): low water
    else {
      baseWL = Math.random() * 60;   // 0-60 cm
      baseRain = Math.random() * 10;
    }

    // Some natural jitter/decay for realism
    const last = state && state.prevSensorData && state.prevSensorData[node];
    const smoothWL = last
      ? last.water_level + (baseWL - last.water_level) * (0.4 + 0.3 * Math.random())
      : baseWL;

    // Other features: humidity usually 60-95, cloud 30-100
    snapshot[node] = {
      rainfall:      parseFloat((baseRain + Math.random() * 2).toFixed(2)),
      humidity:      parseFloat((62 + Math.random() * 32).toFixed(2)),
      cloud_density: parseFloat((35 + Math.random() * 60).toFixed(2)),
      water_level:   parseFloat((smoothWL).toFixed(2)),
      rate_of_rise:  parseFloat((preds[node] === "RED"
                                  ? 0.4 + Math.random() * 0.3
                                  : preds[node] === "YELLOW"
                                    ? 0.15 + Math.random() * 0.2
                                    : Math.random() * 0.15
                                 ).toFixed(3)),
    };
  });
  return snapshot;
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
    let sensorData = await fetchLatestSensorData();
    const sosData = await fetchLatestSosData();
    const hasUsableSensorData = !isPlaceholderSnapshot(sensorData);

    // Fall back to the normal simulated dashboard behavior when the backend
    // is unreachable or only serving placeholder/default snapshots.
    if (!sensorData || !hasUsableSensorData) sensorData = generateMockSnapshot();
    if (!preds || !hasUsableSensorData) preds = generateMockPredictions();

    // Enforce water level thresholds on the frontend
    if (sensorData) {
      Object.keys(sensorData).forEach(node => {
        const wl = sensorData[node].water_level;
        if (wl !== undefined) {
          if (wl <= 60) preds[node] = "GREEN";
          else if (wl <= 90) preds[node] = "YELLOW";
          else preds[node] = "RED";
        }
      });
    }

    state.prevSensorData = state.latestSensorData;
    state.latestSensorData = sensorData;
    state.latestSosData = sosData;
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

  checkHealth().then(ok => setNavStatus(ok));
});