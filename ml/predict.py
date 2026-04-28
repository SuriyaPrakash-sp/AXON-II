"""
predict.py — load trained model and run inference

Usage (standalone):
    cd ml/
    python predict.py

Or import:
    from predict import load_model, run_inference
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model import FloodGCNLSTM
from preprocess import load_adjacency
from utils import (
    NODE_ORDER, NODE_TO_IDX, NUM_NODES,
    FEATURE_COLS, NUM_FEATURES, SEQ_LEN,
    COLOR_MAP, normalize_adjacency,
    node_predictions_to_colors,
    apply_normalizer,                  # now lives in utils — import works
)

MODEL_PATH = Path(__file__).parent / "model.pth"


# ──────────────────────────────────────────────
# 1. Load model (called once at startup)
# ──────────────────────────────────────────────

def load_model(model_path=MODEL_PATH, device=None):
    """
    Load model.pth and return (model, adj_norm, norm_params, device).

    Returns:
        model       : FloodGCNLSTM in eval mode
        adj_norm    : torch.FloatTensor (N, N)
        norm_params : dict {"f_min": np.ndarray, "f_max": np.ndarray}
        device      : torch.device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    hp = checkpoint["hyperparams"]

    model = FloodGCNLSTM(
        num_nodes    = hp["num_nodes"],
        num_features = hp["num_features"],
        gcn_hidden   = hp["gcn_hidden"],
        lstm_hidden  = hp["lstm_hidden"],
        num_classes  = hp["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Adjacency
    adj_raw  = load_adjacency()
    adj_norm = normalize_adjacency(adj_raw).to(device)

    # Normalisation params saved during training
    raw_np = checkpoint["norm_params"]
    norm_params = {
        "f_min": np.array(raw_np["f_min"], dtype=np.float32),
        "f_max": np.array(raw_np["f_max"], dtype=np.float32),
    }

    print(f"[predict] Model loaded from {model_path}  (device={device})")
    return model, adj_norm, norm_params, device


# ──────────────────────────────────────────────
# 2. Run inference
# ──────────────────────────────────────────────

def run_inference(raw_input, model, adj_norm, norm_params, device):
    """
    Run model inference on a raw input window.

    Args:
        raw_input : one of:
          - np.ndarray (SEQ_LEN, N, F)  — used directly
          - dict  {"N1": {"rainfall": ..., ...}, ...}
                  Interpreted as a single timestep; replicated SEQ_LEN times.
          - list of dicts  — SEQ_LEN sequential snapshots.
                  Shorter lists are front-padded; longer lists are trimmed.

        model       : loaded FloodGCNLSTM
        adj_norm    : normalised adjacency on the correct device
        norm_params : dict with "f_min" and "f_max"
        device      : torch.device

    Returns:
        dict  {"N1": "GREEN", "N2": "YELLOW", ...}
    """
    # ── Parse input into (SEQ_LEN, N, F) numpy array ──────────────────
    if isinstance(raw_input, np.ndarray):
        features = raw_input.astype(np.float32)

    else:
        # Normalise to list of dicts
        if isinstance(raw_input, dict):
            snapshots = [raw_input] * SEQ_LEN
        else:
            snapshots = list(raw_input)
            if len(snapshots) < SEQ_LEN:
                pad = [snapshots[0]] * (SEQ_LEN - len(snapshots))
                snapshots = pad + snapshots
            snapshots = snapshots[-SEQ_LEN:]

        features = np.zeros((SEQ_LEN, NUM_NODES, NUM_FEATURES), dtype=np.float32)
        for t, snapshot in enumerate(snapshots):
            for node_id, vals in snapshot.items():
                if node_id not in NODE_TO_IDX:
                    continue
                nidx = NODE_TO_IDX[node_id]
                for f_idx, col in enumerate(FEATURE_COLS):
                    features[t, nidx, f_idx] = float(vals.get(col, 0.0))

    # ── Normalise ───────────────────────────────────────────────────────
    features = apply_normalizer(features, norm_params["f_min"], norm_params["f_max"])
    features = np.clip(features, 0.0, 1.0)

    # ── Tensor: (1, SEQ_LEN, N, F) ─────────────────────────────────────
    x = torch.FloatTensor(features).unsqueeze(0).to(device)

    # ── Inference ───────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(x, adj_norm)          # (1, N, C)
        pred   = logits.argmax(dim=-1)[0]    # (N,)

    return node_predictions_to_colors(pred.cpu().numpy())


# ──────────────────────────────────────────────
# 3. Quick standalone test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    if not MODEL_PATH.exists():
        print(f"No model found at {MODEL_PATH}. Run train.py first.")
        sys.exit(1)

    model, adj_norm, norm_params, device = load_model()

    dummy_snapshot = {
        node: {col: 0.5 for col in FEATURE_COLS}
        for node in NODE_ORDER
    }
    result = run_inference(dummy_snapshot, model, adj_norm, norm_params, device)
    print("\nSample prediction:")
    for node, color in sorted(result.items(), key=lambda kv: int(kv[0][1:])):
        print(f"  {node}: {color}")