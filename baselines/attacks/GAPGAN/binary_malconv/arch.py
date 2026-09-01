import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# MODEL CONFIGURATION
# Aligned to GAPGAN paper (ECAI 2020) §4.2:
#   "MalConv first embeds each byte in input binaries to
#    8-dimensional vector, then uses two convolution layers
#    with different activation functions for classification."
#
# Key params: channels=128, window_size=500, out_size=1
# Loss: BCELoss (binary, sigmoid output)
# Input bytes: 0-255 with literal zero padding (paper-truth)
#
# Layer names MUST match detector/target_model.py exactly:
#   embedding, conv1, conv2, fc1, fc2
# ==========================================

class MalConv(nn.Module):
    # trained to minimize BCELoss (sigmoid output)
    # criterion = nn.BCELoss()
    def __init__(
        self,
        input_length=2_000_000,
        channels=128,
        window_size=500,
        embedding_dim=8,
    ):
        super(MalConv, self).__init__()
        self.input_length = input_length
        # Paper-truth input domain: raw bytes 0-255, zero padding stays byte 0.
        self.embedding = nn.Embedding(256, embedding_dim)
        self.window_size = window_size

        # Two gated convolution layers (paper §4.2)
        self.conv1 = nn.Conv1d(embedding_dim, channels,
                               kernel_size=window_size, stride=window_size)
        self.conv2 = nn.Conv1d(embedding_dim, channels,
                               kernel_size=window_size, stride=window_size)

        # Fully-connected classification head
        self.fc1 = nn.Linear(channels, channels)
        self.fc2 = nn.Linear(channels, 1)

    def forward(self, x):
        # Embedding: (batch, t) -> (batch, t, emb_dim)
        emb = self.embedding(x.long())
        # Transpose for Conv1d: (batch, emb_dim, t)
        emb = emb.permute(0, 2, 1)

        # Gating mechanism (as in original MalConv)
        gate = torch.sigmoid(self.conv1(emb))
        feat = F.relu(self.conv2(emb))
        x = gate * feat  # element-wise gating

        # Global max pooling
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (batch, channels)

        # Classification head
        x = F.relu(self.fc1(x))
        x = self.fc2(x).squeeze(-1)
        prob = torch.sigmoid(x)  # (batch,) — probability of benign
        return prob

    def predict(self, x):
        prob = self.forward(x)
        return torch.where(
            prob >= 0.5,
            torch.ones_like(prob),
            -torch.ones_like(prob),
        )
