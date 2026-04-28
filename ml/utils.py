"""
utils.py — shared constants, node mapping, label helpers
"""

import numpy as np
import torch

# Canonical node order — all files must use this
NODE_ORDER = [f"N{i}" for i in range(1, 16)]  # ['N1', ..., 'N15']
NODE_TO_IDX = {n: i for i, n in enumerate(NODE_ORDER)}
NUM_NODES = 15

# Feature columns in the order they appear in dataset.json
FEATURE_COLS = ["rainfall", "humidity", "cloud_density", "water_level", "rate_of_rise"]
NUM_FEATURES = len(FEATURE_COLS)  # 5

# Risk label mapping  (flood_risk in dataset.json is already 0/1/2)
LABEL_MAP     = {0: "SAFE", 1: "WARNING", 2: "FLOOD"}
COLOR_MAP     = {0: "GREEN", 1: "YELLOW",  2: "RED"}
REVERSE_LABEL = {"SAFE": 0, "WARNING": 1, "FLOOD": 2,
                 0: 0,       1: 1,         2: 2}      # accept int keys too

# Sequence length for sliding window
SEQ_LEN = 6

# Model hyperparameters (shared between train.py and predict.py)
GCN_HIDDEN  = 32
LSTM_HIDDEN = 64
NUM_CLASSES = 3


# ──────────────────────────────────────────────
# Feature normalisation helpers
# ──────────────────────────────────────────────

def apply_normalizer(features: np.ndarray,
                     f_min:    np.ndarray,
                     f_max:    np.ndarray) -> np.ndarray:
    """
    Min-max scale features to [0, 1].

    Args:
        features : np.ndarray (..., F)
        f_min    : np.ndarray (F,)
        f_max    : np.ndarray (F,)

    Returns:
        np.ndarray same shape as features, dtype float32
    """
    return ((features - f_min) / (f_max - f_min)).astype(np.float32)


# ──────────────────────────────────────────────
# Prediction helpers
# ──────────────────────────────────────────────

def node_predictions_to_colors(pred_indices):
    """
    Convert array of predicted class indices (length 15) to
    a dict of {node_name: color_string}.

    Args:
        pred_indices: list or np.array of length 15, values in {0, 1, 2}

    Returns:
        dict e.g. {"N1": "GREEN", "N2": "YELLOW", ...}
    """
    return {NODE_ORDER[i]: COLOR_MAP[int(pred_indices[i])] for i in range(NUM_NODES)}


# ──────────────────────────────────────────────
# Graph helpers
# ──────────────────────────────────────────────

def normalize_adjacency(adj: np.ndarray) -> torch.Tensor:
    """
    Symmetric normalisation: Â = D^{-1/2} (A + I) D^{-1/2}

    Args:
        adj: np.ndarray (N, N) raw adjacency matrix

    Returns:
        torch.FloatTensor (N, N) normalised adjacency
    """
    adj = adj + np.eye(adj.shape[0])           # add self-loops
    degree = adj.sum(axis=1)                    # row-wise degree
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0     # guard against isolated nodes
    D = np.diag(d_inv_sqrt)
    adj_norm = D @ adj @ D
    return torch.FloatTensor(adj_norm)


# ──────────────────────────────────────────────
# Debug helper
# ──────────────────────────────────────────────

def print_shapes(X: torch.Tensor, y: torch.Tensor):
    """Pretty-print tensor shapes for debugging."""
    print(f"  X shape : {tuple(X.shape)}  (samples, seq_len, nodes, features)")
    print(f"  y shape : {tuple(y.shape)}  (samples, nodes)")
    print(f"  Classes : {torch.unique(y).tolist()}")