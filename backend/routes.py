"""
routes.py — Flask Blueprint with POST /data and GET /predict
"""

import threading
from flask import Blueprint, request, jsonify
from model_loader import get_model_bundle

api = Blueprint("api", __name__)

# Thread-safe store for the latest input window
_lock        = threading.Lock()
_latest_data = None   # list of snapshots (most recent SEQ_LEN kept)

SEQ_LEN = 6


# ──────────────────────────────────────────────
# POST /data
# ──────────────────────────────────────────────

@api.route("/data", methods=["POST"])
def receive_data():
    """
    Accept a new sensor snapshot and append to the rolling window.

    Expected JSON body (single timestep):
    {
      "N1": {"rainfall": 12.3, "humidity": 0.8, "cloud_density": 0.6,
             "water_level": 1.2, "rate_of_rise": 0.05},
      "N2": { ... },
      ...
    }

    Or a list of timesteps:
    [
      { "N1": {...}, ... },
      ...
    ]

    Returns:
      200 {"status": "ok", "window_size": N}
      400 if body is missing or malformed
    """
    global _latest_data

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    with _lock:
        if _latest_data is None:
            _latest_data = []

        if isinstance(payload, dict):
            # Single snapshot
            _latest_data.append(payload)
        elif isinstance(payload, list):
            _latest_data.extend(payload)
        else:
            return jsonify({"error": "Body must be a JSON object or array"}), 400

        # Keep only the last SEQ_LEN snapshots
        _latest_data = _latest_data[-SEQ_LEN:]

    return jsonify({"status": "ok", "window_size": len(_latest_data)}), 200


# ──────────────────────────────────────────────
# GET /predict
# ──────────────────────────────────────────────

@api.route("/predict", methods=["GET"])
def predict():
    """
    Run model on the latest stored data window.

    Returns:
      200 {
            "predictions": {"N1": "GREEN", "N2": "RED", ...},
            "window_size": 6
          }
      503 if no data has been POSTed yet
    """
    with _lock:
        data_snapshot = list(_latest_data) if _latest_data else None

    if not data_snapshot:
        return jsonify({
            "error": "No data available. POST to /data first.",
            "predictions": None,
        }), 503

    try:
        bundle = get_model_bundle()
        predictions = bundle.predict(data_snapshot)
    except Exception as exc:
        return jsonify({"error": f"Inference failed: {str(exc)}"}), 500

    return jsonify({
        "predictions": predictions,
        "window_size": len(data_snapshot),
    }), 200


# ──────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────

@api.route("/health", methods=["GET"])
def health():
    """Simple liveness check."""
    with _lock:
        has_data = _latest_data is not None and len(_latest_data) > 0

    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "has_data": has_data,
    }), 200