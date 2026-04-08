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
    apply_normalizer,
)

MODEL_PATH = Path(__file__).parent / "model.pth"

# ──────────────────────────────────────────────
# 1. Load model (called once at startup)
# ──────────────────────────────────────────────

def load_model(model_path=MODEL_PATH, device=None):
    """
    Load model.pth and return (model, adj_norm, norm_params, device).

    Returns:
        model      : FloodGCNLSTM in eval mode
        adj_norm   : torch.FloatTensor (N, N)
        norm_params: dict {"f_min": np.array, "f_max": np.array}
        device     : torch.device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)
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

    # Normalisation params
    raw_np = checkpoint["norm_params"]
    norm_params = {
        "f_min": np.array(raw_np["f_min"], dtype=np.float32),
        "f_max": np.array(raw_np["f_max"], dtype=np.float32),
    }

    print(f"Model loaded from {model_path}  (device={device})")
    return model, adj_norm, norm_params, device


# ──────────────────────────────────────────────
# 2. Run inference
# ──────────────────────────────────────────────

def run_inference(raw_input, model, adj_norm, norm_params, device):
    """
    Run model inference on a raw input window.

    Args:
        raw_input : dict OR list of dicts OR np.ndarray (SEQ_LEN, N, F)
                    If dict: {"N1": {"rainfall": ..., ...}, ...}
                      — interpreted as a single timestep; window is
                        SEQ_LEN copies of this snapshot.
                    If list of dicts: SEQ_LEN sequential snapshots.
                    If np.ndarray (SEQ_LEN, N, F): used directly.

        model      : loaded FloodGCNLSTM
        adj_norm   : normalised adjacency on correct device
        norm_params: dict with "f_min" and "f_max"
        device     : torch.device

    Returns:
        dict  {"N1": "GREEN", "N2": "YELLOW", ...}
    """
    # ── Parse input into (SEQ_LEN, N, F) numpy array ──────────────────
    if isinstance(raw_input, np.ndarray):
        features = raw_input.astype(np.float32)   # assume already (T, N, F)
    else:
        # Normalise to list of dicts
        if isinstance(raw_input, dict):
            snapshots = [raw_input] * SEQ_LEN  # replicate single snapshot
        else:
            snapshots = list(raw_input)
            if len(snapshots) < SEQ_LEN:
                # Pad at the front by repeating first snapshot
                pad = [snapshots[0]] * (SEQ_LEN - len(snapshots))
                snapshots = pad + snapshots
            snapshots = snapshots[-SEQ_LEN:]   # take last SEQ_LEN

        features = np.zeros((SEQ_LEN, NUM_NODES, NUM_FEATURES), dtype=np.float32)
        for t, snapshot in enumerate(snapshots):
            # snapshot is {"N1": {feat: val, ...}, "N2": {...}, ...}
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

    # Dummy input: all nodes with mid-range feature values
    dummy_snapshot = {
        node: {col: 0.5 for col in FEATURE_COLS}
        for node in NODE_ORDER
    }
    result = run_inference(dummy_snapshot, model, adj_norm, norm_params, device)
    print("\nSample prediction:")
    for node, color in sorted(result.items(), key=lambda x: int(x[0][1:])):
        print(f"  {node}: {color}")