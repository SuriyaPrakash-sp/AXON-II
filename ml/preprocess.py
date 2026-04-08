"""
preprocess.py — load dataset.json + adjacency.json, build tensors

Output tensors:
  X : (samples, SEQ_LEN, NUM_NODES, NUM_FEATURES)  float32
  y : (samples, NUM_NODES)                          long
  adj_norm : (NUM_NODES, NUM_NODES)                 float32  (normalised)
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
)

DATA_DIR = Path(__file__).parent.parent / "data"


# ──────────────────────────────────────────────
# 1. Load raw JSON files
# ──────────────────────────────────────────────

def load_dataset():
    """
    Load dataset.json.

    Expected format (example):
    [
      {
        "timestep": 0,
        "nodes": {
          "N1": {"rainfall": 12.3, "humidity": 0.8, ..., "flood_risk": "SAFE"},
          "N2": {...},
          ...
        }
      },
      ...
    ]

    Returns:
        features : np.ndarray (T, N, F)  float32
        targets  : np.ndarray (T, N)     int
    """
    with open(DATA_DIR / "dataset.json") as f:
        raw = json.load(f)

    # Sort timesteps just in case they are unordered
    raw = sorted(raw, key=lambda x: x["timestep"])
    T = len(raw)

    features = np.zeros((T, NUM_NODES, NUM_FEATURES), dtype=np.float32)
    targets  = np.zeros((T, NUM_NODES), dtype=np.int64)

    for t, record in enumerate(raw):
        nodes = record["nodes"]
        for node_id, vals in nodes.items():
            if node_id not in NODE_TO_IDX:
                continue
            idx = NODE_TO_IDX[node_id]
            for f_idx, col in enumerate(FEATURE_COLS):
                features[t, idx, f_idx] = float(vals[col])
            targets[t, idx] = REVERSE_LABEL[vals["flood_risk"]]

    return features, targets


def load_adjacency():
    """
    Load adjacency.json.

    Expected format:
    {
      "edges": ["N1->N3", "N3->N5", ...]
    }
    or simply a list: ["N1->N3", ...]

    Returns:
        adj : np.ndarray (N, N)  binary directed adjacency matrix
    """
    with open(DATA_DIR / "adjacency.json") as f:
        raw = json.load(f)

    # Accept both {"edges": [...]} and plain [...]
    edges = raw["edges"] if isinstance(raw, dict) else raw

    adj = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)
    for edge in edges:
        src, dst = edge.split("->")
        src, dst = src.strip(), dst.strip()
        if src in NODE_TO_IDX and dst in NODE_TO_IDX:
            adj[NODE_TO_IDX[src], NODE_TO_IDX[dst]] = 1.0

    return adj


# ──────────────────────────────────────────────
# 2. Normalise features (per-feature, over all T and N)
# ──────────────────────────────────────────────

def fit_normalizer(features):
    """
    Compute per-feature min/max over the entire dataset.

    Args:
        features: np.ndarray (T, N, F)

    Returns:
        f_min, f_max: np.ndarray (F,)
    """
    f_min = features.reshape(-1, features.shape[-1]).min(axis=0)  # (F,)
    f_max = features.reshape(-1, features.shape[-1]).max(axis=0)
    # Avoid divide-by-zero for constant features
    f_max[f_max == f_min] = f_min[f_max == f_min] + 1.0
    return f_min, f_max


def apply_normalizer(features, f_min, f_max):
    return (features - f_min) / (f_max - f_min)


# ──────────────────────────────────────────────
# 3. Build sliding-window sequences
# ──────────────────────────────────────────────

def make_sequences(features, targets, seq_len=SEQ_LEN):
    """
    Create overlapping windows of length seq_len.

    features : (T, N, F)
    targets  : (T, N)

    Returns:
        X : (samples, seq_len, N, F)  — input window
        y : (samples, N)              — label at last timestep of window
    """
    T = features.shape[0]
    samples = T - seq_len  # predict step t using steps [t-seq_len .. t-1]

    X_list, y_list = [], []
    for i in range(samples):
        X_list.append(features[i : i + seq_len])   # (seq_len, N, F)
        y_list.append(targets[i + seq_len])         # (N,)

    X = np.stack(X_list, axis=0)  # (samples, seq_len, N, F)
    y = np.stack(y_list, axis=0)  # (samples, N)
    return X, y


# ──────────────────────────────────────────────
# 4. Master pipeline
# ──────────────────────────────────────────────

def preprocess(seq_len=SEQ_LEN):
    """
    Full pipeline: load → normalise → window → tensor.

    Returns:
        X        : torch.FloatTensor (samples, seq_len, N, F)
        y        : torch.LongTensor  (samples, N)
        adj_norm : torch.FloatTensor (N, N)
        norm_params : dict {"f_min": ..., "f_max": ...} for inference
    """
    print("Loading dataset …")
    features, targets = load_dataset()
    print(f"  Raw features : {features.shape}   targets : {targets.shape}")

    print("Loading adjacency …")
    adj = load_adjacency()
    adj_norm = normalize_adjacency(adj)
    print(f"  Adjacency    : {adj.shape}  edges={int(adj.sum())}")

    print("Normalising features …")
    f_min, f_max = fit_normalizer(features)
    features_norm = apply_normalizer(features, f_min, f_max)

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