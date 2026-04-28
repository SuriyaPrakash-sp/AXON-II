"""
routes.py — Flask Blueprint

Structured ESP32 ingestion with dataset-backed fallback:
  - /data accepts validated N3 sensor packets
  - /sos_location accepts validated SOS GPS packets
  - N1..N2 and N4..N15 always come from data/dataset.json snapshots
  - N3 is overridden only with valid, fresh ESP32 data
"""

import json
import threading
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from flask import Blueprint, jsonify, request

api = Blueprint("api", __name__)

_lock = threading.Lock()

SEQ_LEN = 6
LIVE_SENSOR_TIMEOUT_SEC = 15
NODES = [f"N{i}" for i in range(1, 16)]
SAFE_MAX_WL = 25.0
WARNING_MAX_WL = 35.0

_latest_data = []
_dataset_cursor = 0

_last_n3_seen_at = None
_last_n3_packet = {
    "node_id": "N3",
    "water_level": None,
    "timestamp": None,
    "rate_of_rise": None,
}

_last_sos_seen_at = None
_last_sos = {
    "type": "sos_location",
    "node_id": "SOS1",
    "mapped_node": "N3",
    "lat": None,
    "lon": None,
    "battery": None,
    "active": False,
    "timestamp": None,
}


def _load_dataset_snapshots():
    dataset_path = Path(__file__).parent.parent / "data" / "dataset.json"
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    records = payload.get("data", [])
    snapshots = []

    for rec in records:
        nodes = {}
        for nd in rec.get("nodes", []):
            nid = nd.get("node_id")
            if nid not in NODES:
                continue
            nodes[nid] = {
                "rainfall": float(nd.get("rainfall", 0.0)),
                "humidity": float(nd.get("humidity", 50.0)),
                "cloud_density": float(nd.get("cloud_density", 50.0)),
                "water_level": float(nd.get("water_level", 0.0)),
                "rate_of_rise": float(nd.get("rate_of_rise", 0.0)),
            }
        if all(n in nodes for n in NODES):
            snapshots.append(nodes)

    if not snapshots:
        raise RuntimeError("dataset.json has no usable snapshots")
    return snapshots


_dataset_snapshots = _load_dataset_snapshots()


def _is_live(ts):
    return ts is not None and (time.time() - ts) <= LIVE_SENSOR_TIMEOUT_SEC


def _next_dataset_snapshot():
    global _dataset_cursor
    snap = deepcopy(_dataset_snapshots[_dataset_cursor % len(_dataset_snapshots)])
    _dataset_cursor = (_dataset_cursor + 1) % len(_dataset_snapshots)
    return snap


def _append_snapshot(snapshot):
    _latest_data.append(snapshot)
    _latest_data[:] = _latest_data[-SEQ_LEN:]


def _seed_window_if_needed():
    while len(_latest_data) < SEQ_LEN:
        _append_snapshot(_next_dataset_snapshot())


def _parse_iso_ts(ts):
    if not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _coerce_float(payload, key):
    try:
        return float(payload[key])
    except (TypeError, ValueError, KeyError):
        raise ValueError(f"Field '{key}' must be a number")


def _validate_sensor_payload(payload):
    if not isinstance(payload, dict):
        return "JSON body must be an object"

    required = ["type", "node_id", "water_level", "rate_of_rise"]
    missing = [k for k in required if k not in payload]
    if missing:
        return f"Missing required field(s): {', '.join(missing)}"

    if payload.get("node_id") != "N3":
        return "Only node_id 'N3' is accepted on /data"

    if payload.get("type") != "sensor":
        return "Field 'type' must be 'sensor'"

    for key in ["water_level", "rate_of_rise"]:
        try:
            _coerce_float(payload, key)
        except ValueError as exc:
            return str(exc)

    return None


def _validate_sos_payload(payload):
    if not isinstance(payload, dict):
        return "JSON body must be an object"

    required = ["node_id", "lat", "lon"]
    missing = [k for k in required if k not in payload]
    if missing:
        return f"Missing required field(s): {', '.join(missing)}"

    if payload.get("node_id") != "SOS1":
        return "Only node_id 'SOS1' is accepted on /sos_location"

    try:
        lat = _coerce_float(payload, "lat")
        lon = _coerce_float(payload, "lon")
    except ValueError as exc:
        return str(exc)

    if not (-90 <= lat <= 90):
        return "Field 'lat' out of range (-90..90)"
    if not (-180 <= lon <= 180):
        return "Field 'lon' out of range (-180..180)"

    if "active" in payload and not isinstance(payload["active"], bool):
        return "Field 'active' must be boolean"

    return None


def _build_snapshot_with_n3_override(sensor_payload):
    global _last_n3_seen_at, _last_n3_packet

    base = _next_dataset_snapshot()
    n3_base = base["N3"]

    wl = _coerce_float(sensor_payload, "water_level")
    ror = _coerce_float(sensor_payload, "rate_of_rise")

    ts = sensor_payload.get("timestamp", datetime.utcnow().isoformat())

    base["N3"] = {
        # Keep dataset-generated companion features for N3.
        "rainfall": n3_base["rainfall"],
        "humidity": n3_base["humidity"],
        "cloud_density": n3_base["cloud_density"],
        "water_level": wl,
        "rate_of_rise": ror,
    }

    _last_n3_packet = {
        "node_id": "N3",
        "water_level": wl,
        "timestamp": ts,
        "rate_of_rise": ror,
    }
    _last_n3_seen_at = time.time()
    return base


@api.route("/data", methods=["POST"])
def receive_data():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    err = _validate_sensor_payload(payload)
    if err:
        return jsonify({"error": err}), 400

    with _lock:
        _seed_window_if_needed()
        snap = _build_snapshot_with_n3_override(payload)
        _append_snapshot(snap)

    return jsonify({"status": "ok", "accepted_node": "N3", "window_size": len(_latest_data)}), 200


@api.route("/sensor/N3", methods=["POST"])
def receive_n3_sensor():
    return receive_data()


@api.route("/sos_location", methods=["POST"])
def receive_sos_location():
    global _last_sos_seen_at, _last_sos

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    err = _validate_sos_payload(payload)
    if err:
        return jsonify({"error": err}), 400

    with _lock:
        _last_sos = {
            "type": "sos_location",
            "node_id": "SOS1",
            "mapped_node": "N3",
            "lat": float(payload["lat"]),
            "lon": float(payload["lon"]),
            "battery": float(payload["battery"]) if "battery" in payload else _last_sos.get("battery"),
            "active": bool(payload.get("active", True)),
            "timestamp": payload.get("timestamp", datetime.utcnow().isoformat()),
        }
        _last_sos_seen_at = time.time()

    return jsonify({"status": "ok", "mapped_node": "N3"}), 200


@api.route("/predict", methods=["GET"])
def predict():
    with _lock:
        _seed_window_if_needed()

        next_snap = _next_dataset_snapshot()
        if _is_live(_last_n3_seen_at) and _last_n3_packet["water_level"] is not None:
            next_snap["N3"] = {
                "rainfall": next_snap["N3"]["rainfall"],
                "humidity": next_snap["N3"]["humidity"],
                "cloud_density": next_snap["N3"]["cloud_density"],
                "water_level": _last_n3_packet["water_level"],
                "rate_of_rise": _last_n3_packet["rate_of_rise"] if _last_n3_packet["rate_of_rise"] is not None else next_snap["N3"]["rate_of_rise"],
            }

        _append_snapshot(next_snap)
        data_snapshot = list(_latest_data)
        latest_snapshot = data_snapshot[-1]

    # Use the same thresholding logic as generate_dataset.py to keep behavior
    # stable and explainable across all nodes.
    predictions = {}
    for node in NODES:
        wl = float(latest_snapshot[node]["water_level"])
        if wl < SAFE_MAX_WL:
            predictions[node] = "GREEN"
        elif wl < WARNING_MAX_WL:
            predictions[node] = "YELLOW"
        else:
            predictions[node] = "RED"

    return jsonify({"predictions": predictions, "window_size": len(data_snapshot)}), 200


@api.route("/latest", methods=["GET"])
def latest():
    with _lock:
        _seed_window_if_needed()
        snapshot = deepcopy(_latest_data[-1])
    return jsonify({"nodes": snapshot}), 200


@api.route("/sos/latest", methods=["GET"])
def latest_sos():
    with _lock:
        sos = deepcopy(_last_sos)
        is_live = _is_live(_last_sos_seen_at)

    sos["is_live"] = is_live
    if not is_live:
        sos["active"] = False
        sos["lat"] = None
        sos["lon"] = None
    return jsonify(sos), 200


@api.route("/health", methods=["GET"])
def health():
    with _lock:
        _seed_window_if_needed()
        win_size = len(_latest_data)

    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "has_data": win_size > 0,
        "window_size": win_size,
        "dataset_snapshots": len(_dataset_snapshots),
        "n3_live": _is_live(_last_n3_seen_at),
        "has_sos": _is_live(_last_sos_seen_at),
    }), 200