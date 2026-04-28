"""
preprocess.py — load dataset.json + adjacency.json, build tensors

Output tensors:
  X        : (samples, SEQ_LEN, NUM_NODES, NUM_FEATURES)  float32
  y        : (samples, NUM_NODES)                          long
  adj_norm : (NUM_NODES, NUM_NODES)                        float32  (normalised)
"""

import json
import numpy as np
import torch
from pathlib import Path

from utils import (
    NODE_ORDER, NODE_TO_IDX, NUM_NODES,
    FEATURE_COLS, NUM_FEATURES,
    REVERSE_LABEL, SEQ_LEN,
    normalize_adjacency, print_shapes,
    apply_normalizer,
)

DATA_DIR = Path(__file__).parent.parent / "data"


# ──────────────────────────────────────────────
# 1. Load raw JSON files
# ──────────────────────────────────────────────

def load_dataset():
    """
    Load dataset.json.

    Supports two formats:

    Format A — timeseries list at top level:
      [ {"timestep": 0, "nodes": {"N1": {...}, ...}}, ... ]

    Format B — Anthropic-generated format (what your dataset.json uses):
      {
        "metadata": {...},
        "graph": {...},
        "data": [
          {
            "time_step": 0,
            "nodes": [
              {"node_id": "N1", "rainfall": ..., "flood_risk": 0},
              ...
            ]
          },
          ...
        ]
      }

    In Format B, flood_risk is already an integer (0/1/2).

    Returns:
        features : np.ndarray (T, N, F)  float32
        targets  : np.ndarray (T, N)     int64
    """
    with open(DATA_DIR / "dataset.json") as f:
        raw = json.load(f)

    # Detect format
    if isinstance(raw, list):
        # Format A: plain list of timestep records
        records = sorted(raw, key=lambda x: x["timestep"])
        use_format_b = False
    else:
        # Format B: dict with "data" key
        records = sorted(raw["data"], key=lambda x: x["time_step"])
        use_format_b = True

    T = len(records)
    features = np.zeros((T, NUM_NODES, NUM_FEATURES), dtype=np.float32)
    targets  = np.zeros((T, NUM_NODES), dtype=np.int64)

    for t, record in enumerate(records):
        if use_format_b:
            # nodes is a list of dicts with "node_id" key
            nodes_iter = {n["node_id"]: n for n in record["nodes"]}.items()
        else:
            # nodes is a plain dict {"N1": {...}, ...}
            nodes_iter = record["nodes"].items()

        for node_id, vals in nodes_iter:
            if node_id not in NODE_TO_IDX:
                continue
            idx = NODE_TO_IDX[node_id]

            for f_idx, col in enumerate(FEATURE_COLS):
                features[t, idx, f_idx] = float(vals.get(col, 0.0))

            risk_val = vals.get("flood_risk", 0)
            # flood_risk may be int (0/1/2) or string ("SAFE"/"WARNING"/"FLOOD")
            targets[t, idx] = REVERSE_LABEL[risk_val]

    return features, targets


def load_adjacency():
    """
    Load adjacency.json.

    Supports:
      {"edges": ["N1->N3", ...]}   or   ["N1->N3", ...]
    Also handles spaces around "->": "N1 -> N3"

    Returns:
        adj : np.ndarray (N, N)  binary directed adjacency matrix
    """
    with open(DATA_DIR / "adjacency.json") as f:
        raw = json.load(f)

    edges = raw["edges"] if isinstance(raw, dict) else raw

    adj = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)
    for edge in edges:
        # Accept both "N1->N3" and "N1 -> N3"
        parts = edge.replace(" ", "").split("->")
        if len(parts) != 2:
            continue
        src, dst = parts
        if src in NODE_TO_IDX and dst in NODE_TO_IDX:
            adj[NODE_TO_IDX[src], NODE_TO_IDX[dst]] = 1.0

    return adj


# ──────────────────────────────────────────────
# 2. Normalise features
# ──────────────────────────────────────────────

def fit_normalizer(features: np.ndarray):
    """
    Compute per-feature min/max over the entire dataset.

    Args:
        features: np.ndarray (T, N, F)

    Returns:
        f_min, f_max: np.ndarray (F,)
    """
    flat   = features.reshape(-1, features.shape[-1])  # (T*N, F)
    f_min  = flat.min(axis=0)
    f_max  = flat.max(axis=0)
    # Avoid divide-by-zero for constant features
    const  = (f_max == f_min)
    f_max[const] = f_min[const] + 1.0
    return f_min, f_max


# ──────────────────────────────────────────────
# 3. Build sliding-window sequences
# ──────────────────────────────────────────────

def make_sequences(features: np.ndarray, targets: np.ndarray, seq_len: int = SEQ_LEN):
    """
    Create overlapping windows of length seq_len.

    features : (T, N, F)
    targets  : (T, N)

    Returns:
        X : (samples, seq_len, N, F)
        y : (samples, N)
    """
    T       = features.shape[0]
    samples = T - seq_len

    if samples <= 0:
        raise ValueError(
            f"Dataset has only {T} timesteps but seq_len={seq_len}. "
            "Need at least seq_len+1 timesteps."
        )

    X_list, y_list = [], []
    for i in range(samples):
        X_list.append(features[i : i + seq_len])
        y_list.append(targets[i + seq_len])

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    return X, y


# ──────────────────────────────────────────────
# 4. Master pipeline
# ──────────────────────────────────────────────

def preprocess(seq_len: int = SEQ_LEN):
    """
    Full pipeline: load → normalise → window → tensor.

    Returns:
        X           : torch.FloatTensor (samples, seq_len, N, F)
        y           : torch.LongTensor  (samples, N)
        adj_norm    : torch.FloatTensor (N, N)
        norm_params : dict {"f_min": np.ndarray, "f_max": np.ndarray}
    """
    print("Loading dataset …")
    features, targets = load_dataset()
    print(f"  Raw features : {features.shape}   targets : {targets.shape}")
    print(f"  Label distribution: { {int(v): int((targets==v).sum()) for v in range(3)} }")

    print("Loading adjacency …")
    adj     = load_adjacency()
    adj_norm = normalize_adjacency(adj)
    print(f"  Adjacency    : {adj.shape}  edges={int(adj.sum())}")

    print("Normalising features …")
    f_min, f_max     = fit_normalizer(features)
    features_norm    = apply_normalizer(features, f_min, f_max)
    features_norm    = np.clip(features_norm, 0.0, 1.0)

    print(f"Building sequences (seq_len={seq_len}) …")
    X_np, y_np = make_sequences(features_norm, targets, seq_len)

    X = torch.FloatTensor(X_np)
    y = torch.LongTensor(y_np)

    print("Final tensors:")
    print_shapes(X, y)

    norm_params = {"f_min": f_min, "f_max": f_max}
    return X, y, adj_norm, norm_params


if __name__ == "__main__":
    X, y, adj_norm, norm_params = preprocess()
    print("\nDone. Shapes verified.")