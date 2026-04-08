"""
model.py — Custom GCN + LSTM flood prediction model

Architecture:
  Input  : (batch, seq_len, N, F)
  GCN    : per-timestep graph convolution → (batch, seq_len, N, gcn_hidden)
  LSTM   : temporal modelling per node    → (batch, N, lstm_hidden)
  Head   : linear classifier              → (batch, N, num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import NUM_NODES, NUM_FEATURES, GCN_HIDDEN, LSTM_HIDDEN, NUM_CLASSES


# ──────────────────────────────────────────────
# 1. Custom GCN Layer
# ──────────────────────────────────────────────

class GCNLayer(nn.Module):
    """
    Single graph convolutional layer.

    Forward: H_out = σ( Â · H_in · W )

    Â is the pre-computed, degree-normalised adjacency matrix
    (with self-loops) — passed in at forward time so it stays
    on the correct device without registering as a parameter.

    Shapes:
        H_in  : (*, N, in_features)
        adj   : (N, N)
        H_out : (*, N, out_features)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : (..., N, in_features)
            adj : (N, N)  normalised adjacency

        Returns:
            out : (..., N, out_features)
        """
        # Linear projection first (cheaper: operate on F before N^2 matmul)
        support = self.linear(x)          # (..., N, out_features)

        # Graph aggregation: Â · support
        # adj is (N, N), support is (..., N, out_features)
        # torch.matmul broadcasts over leading dims
        out = torch.matmul(adj, support)  # (..., N, out_features)

        return F.relu(out)


# ──────────────────────────────────────────────
# 2. Full Model
# ──────────────────────────────────────────────

class FloodGCNLSTM(nn.Module):
    """
    Spatio-temporal flood risk predictor.

    Pipeline:
      1. GCN encodes spatial relationships at each timestep
      2. LSTM captures temporal dynamics per node
      3. Linear head outputs class logits per node

    Input  : (batch, seq_len, N, F)
    Output : (batch, N, num_classes)
    """

    def __init__(
        self,
        num_nodes: int   = NUM_NODES,
        num_features: int= NUM_FEATURES,
        gcn_hidden: int  = GCN_HIDDEN,
        lstm_hidden: int = LSTM_HIDDEN,
        num_classes: int = NUM_CLASSES,
        lstm_layers: int = 1,
        dropout: float   = 0.3,
    ):
        super().__init__()
        self.num_nodes   = num_nodes
        self.gcn_hidden  = gcn_hidden
        self.lstm_hidden = lstm_hidden

        # GCN stack
        self.gcn1 = GCNLayer(num_features, gcn_hidden)
        self.gcn2 = GCNLayer(gcn_hidden,   gcn_hidden)

        self.gcn_dropout = nn.Dropout(dropout)

        # LSTM — applied per node over time
        # Input to LSTM: gcn_hidden per timestep
        self.lstm = nn.LSTM(
            input_size  = gcn_hidden,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            dropout     = dropout if lstm_layers > 1 else 0.0,
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : (batch, seq_len, N, F)
            adj : (N, N)  normalised adjacency (device must match x)

        Returns:
            logits : (batch, N, num_classes)
        """
        batch, seq_len, N, F = x.shape

        # ── Step 1: GCN over each timestep ──────────────────────────────
        # Process all timesteps in one batch by merging batch & time dims
        x_flat = x.view(batch * seq_len, N, F)          # (B*T, N, F)

        gcn_out = self.gcn1(x_flat, adj)                 # (B*T, N, gcn_hidden)
        gcn_out = self.gcn_dropout(gcn_out)
        gcn_out = self.gcn2(gcn_out, adj)                # (B*T, N, gcn_hidden)

        # Restore time dimension
        gcn_out = gcn_out.view(batch, seq_len, N, self.gcn_hidden)
        # (batch, seq_len, N, gcn_hidden)

        # ── Step 2: LSTM per node over time ─────────────────────────────
        # Permute so nodes are in the batch-like dimension for LSTM
        # (batch, seq_len, N, gcn_hidden) → (batch, N, seq_len, gcn_hidden)
        gcn_out = gcn_out.permute(0, 2, 1, 3)

        # Merge batch and node dims for efficient LSTM processing
        # (batch * N, seq_len, gcn_hidden)
        lstm_in = gcn_out.contiguous().view(batch * N, seq_len, self.gcn_hidden)

        lstm_out, _ = self.lstm(lstm_in)
        # lstm_out : (batch * N, seq_len, lstm_hidden)

        # Take the last timestep's hidden state
        last_hidden = lstm_out[:, -1, :]   # (batch * N, lstm_hidden)

        # ── Step 3: Classify ────────────────────────────────────────────
        logits = self.classifier(last_hidden)   # (batch * N, num_classes)

        # Reshape back to (batch, N, num_classes)
        logits = logits.view(batch, N, -1)

        return logits


# ──────────────────────────────────────────────
# 3. Quick sanity check
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from utils import normalize_adjacency
    import numpy as np

    batch, seq_len = 4, 6
    adj_np = np.eye(NUM_NODES, dtype=np.float32)   # identity as dummy adj
    adj    = normalize_adjacency(adj_np)

    model  = FloodGCNLSTM()
    x      = torch.randn(batch, seq_len, NUM_NODES, NUM_FEATURES)
    out    = model(x, adj)

    print(f"Input  : {tuple(x.shape)}")
    print(f"Output : {tuple(out.shape)}  ← expected ({batch}, {NUM_NODES}, {NUM_CLASSES})")
    assert out.shape == (batch, NUM_NODES, NUM_CLASSES), "Shape mismatch!"
    print("model.py self-test passed.")