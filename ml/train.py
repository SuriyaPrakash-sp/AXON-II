"""
train.py — train the GCN+LSTM model and save model.pth

Run:
    cd ml/
    python train.py
"""

import json
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

# Allow running from ml/ or project root
sys.path.insert(0, str(Path(__file__).parent))

from preprocess import preprocess
from model import FloodGCNLSTM
from utils import GCN_HIDDEN, LSTM_HIDDEN, NUM_CLASSES, NUM_NODES, NUM_FEATURES

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
EPOCHS      = 20
BATCH_SIZE  = 8
LR          = 1e-3
WEIGHT_DECAY= 1e-4
SAVE_PATH   = Path(__file__).parent / "model.pth"

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def train():
    # ── 1. Data ──────────────────────────────────────────────────────────
    X, y, adj_norm, norm_params = preprocess()

    # Simple 80/20 train/val split (no shuffle to preserve temporal order)
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val,   y_val   = X[split:], y[split:]

    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val,   y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    adj = adj_norm.to(device)

    print(f"\nTrain samples: {len(train_ds)}   Val samples: {len(val_ds)}")

    # ── 2. Model ─────────────────────────────────────────────────────────
    model = FloodGCNLSTM(
        num_nodes   = NUM_NODES,
        num_features= NUM_FEATURES,
        gcn_hidden  = GCN_HIDDEN,
        lstm_hidden = LSTM_HIDDEN,
        num_classes = NUM_CLASSES,
        dropout     = 0.3,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # ── 3. Loss & optimiser ──────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    # ── 4. Training loop ─────────────────────────────────────────────────
    best_val_loss = float("inf")
    print(f"\n{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>9}  {'Val Acc':>8}")
    print("─" * 48)

    for epoch in range(1, EPOCHS + 1):
        # — Train —
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)         # (batch, N)

            optimizer.zero_grad()
            logits = model(X_batch, adj)         # (batch, N, C)

            # CrossEntropyLoss expects (batch, C, ...) or flatten
            # Flatten to (batch*N, C) and (batch*N,)
            B, N, C = logits.shape
            loss = criterion(
                logits.view(B * N, C),
                y_batch.view(B * N),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # — Validate —
        model.eval()
        val_loss  = 0.0
        correct   = 0
        total     = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                logits  = model(X_batch, adj)
                B, N, C = logits.shape
                loss = criterion(
                    logits.view(B * N, C),
                    y_batch.view(B * N),
                )
                val_loss += loss.item()
                preds   = logits.argmax(dim=-1)  # (batch, N)
                correct += (preds == y_batch).sum().item()
                total   += preds.numel()

        val_loss /= len(val_loader)
        val_acc   = correct / total if total > 0 else 0.0

        print(f"{epoch:>6}  {train_loss:>11.4f}  {val_loss:>9.4f}  {val_acc:>7.2%}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save checkpoint with everything needed for inference
            torch.save({
                "model_state_dict": model.state_dict(),
                "norm_params"     : {
                    "f_min": norm_params["f_min"].tolist(),
                    "f_max": norm_params["f_max"].tolist(),
                },
                "hyperparams": {
                    "gcn_hidden" : GCN_HIDDEN,
                    "lstm_hidden": LSTM_HIDDEN,
                    "num_classes": NUM_CLASSES,
                    "num_nodes"  : NUM_NODES,
                    "num_features": NUM_FEATURES,
                },
            }, SAVE_PATH)

        scheduler.step()

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {SAVE_PATH}")


if __name__ == "__main__":
    train()