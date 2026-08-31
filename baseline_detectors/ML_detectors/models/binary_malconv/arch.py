import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. MODEL CONFIGURATION
# ==========================================
MALCONV_CFG = {
    "max_len": 2000000,  # UPDATED: Paper states ~2 million steps [cite: 9]
    "window_size": 500,
    "stride": 500,
    "vocab_size": 257,
    "emb_size": 8,
    "channels": 128,
    "num_classes": 1,
}


# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
class MalConvArch(nn.Module):
    def __init__(self, cfg=None):
        super(MalConvArch, self).__init__()
        self.cfg = cfg if cfg else MALCONV_CFG.copy()

        vocab_size = self.cfg["vocab_size"]
        emb_size = self.cfg["emb_size"]
        channels = self.cfg["channels"]
        window = self.cfg["window_size"]
        stride = self.cfg["stride"]
        num_classes = self.cfg["num_classes"]

        self.embed = nn.Embedding(vocab_size, emb_size, padding_idx=vocab_size - 1)

        # Convolutional Layers
        self.conv_1 = nn.Conv1d(emb_size, channels, window, stride=stride, bias=True)
        self.conv_2 = nn.Conv1d(emb_size, channels, window, stride=stride, bias=True)

        # Dense Layers
        self.fc_1 = nn.Linear(channels, channels)
        self.fc_2 = nn.Linear(channels, num_classes)

    def forward(self, x):
        # Embed & Transpose
        x = self.embed(x)
        x = x.transpose(1, 2)

        # Gated Conv Mechanism: A * sigma(B)
        # FIX: Removed ReLU on cnn_value to match Eq in Figure 1 & 6 [cite: 115, 461]
        cnn_value = self.conv_1(x)
        cnn_gate = self.conv_2(x)
        cnn_gate = torch.sigmoid(cnn_gate)

        gated = cnn_value * cnn_gate

        # Global Max Pooling
        x = F.max_pool1d(gated, gated.size(2)).squeeze(2)

        # Penultimate Layer (Hidden State)
        # Paper: DeCov penalizes correlation at the penultimate layer
        hidden = self.fc_1(x)
        hidden = F.relu(hidden)

        # Final Classification
        logits = self.fc_2(hidden)

        # Return both logits (for prediction) and hidden (for DeCov)
        return logits, hidden
