import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al. 2017)
    Used to inject sequence position information into the transformer
    """

    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """Add positional encoding to input sequence"""
        return x + self.pe[:, : x.size(1)]


class SeqConvAttn(nn.Module):
    """
    SeqConvAttn: Sequence-based malware classifier with transformer attention

    Architecture:
      1. Byte Embedding (257 → emb_dim)
      2. 1D Convolution (n_filters kernels, kernel_size, stride=kernel_size)
      3. Positional Encoding (sinusoidal)
      4. Transformer Encoder (n_transformer_blocks)
      5. Max Pooling
      6. Classification Head

    Paper defaults:
      - vocab_size: 257 (256 byte values + padding token)
      - emb_dim: 8
      - n_filters: 128
      - kernel_size: 500
      - n_transformer_blocks: 3 (K=3)
      - n_heads: 8
      - ff_expansion: 4
    """

    def __init__(
        self,
        num_classes=6,
        vocab_size=257,
        emb_dim=8,
        n_filters=128,
        kernel_size=500,
        n_transformer_blocks=3,
        n_heads=8,
        ff_expansion=4,
    ):
        super(SeqConvAttn, self).__init__()

        self.num_classes = num_classes
        self.n_filters = n_filters

        # Layer 1: Byte Embedding
        # Maps each byte value (0-255) + padding token (256) to emb_dim-dimensional vector
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=256)

        # Layer 2: 1D Convolution
        # Input: (batch, seq_len, emb_dim) → (batch, seq_len/kernel_size, n_filters)
        # Uses n_filters kernels of size kernel_size with stride kernel_size for compression
        self.conv1d = nn.Conv1d(
            in_channels=emb_dim,
            out_channels=n_filters,
            kernel_size=kernel_size,
            stride=kernel_size,
        )

        # Layer 3: Positional Encoding
        # Max sequence length after conv: 400000/500 = 800
        self.pos_encoding = PositionalEncoding(d_model=n_filters, max_len=1000)

        # Layer 4: Transformer Encoder Blocks
        # Standard transformer encoder with multihead attention + feedforward
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_filters,  # 128
            nhead=n_heads,  # 8 heads
            dim_feedforward=n_filters * ff_expansion,  # 512 (expansion factor 4)
            batch_first=True,
            dropout=0.0,  # Paper doesn't specify dropout in transformer
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_blocks
        )

        # Layer 5: Classification Head
        # Max pool → FC(n_filters→n_filters) → FC(n_filters→num_classes)
        self.fc = nn.Sequential(
            nn.Linear(n_filters, n_filters),
            nn.ReLU(),
            nn.Linear(n_filters, num_classes),
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: (batch_size, seq_len) - byte sequences, values 0-256

        Returns:
            logits: (batch_size, num_classes)
        """
        # Layer 1: Embedding
        # (batch, 400000) → (batch, 400000, emb_dim)
        x = self.embedding(x)

        # Layer 2: 1D Convolution
        # (batch, 400000, emb_dim) → permute → (batch, emb_dim, 400000)
        x = x.permute(0, 2, 1)
        # (batch, emb_dim, 400000) → Conv1d → (batch, n_filters, seq_len/kernel_size)
        x = self.conv1d(x)
        # (batch, n_filters, seq_len/kernel_size) → permute → (batch, seq_len/kernel_size, n_filters)
        x = x.permute(0, 2, 1)

        # Layer 3: Positional Encoding
        # (batch, seq_len/kernel_size, n_filters) + PE → (batch, seq_len/kernel_size, n_filters)
        x = self.pos_encoding(x)

        # Layer 4: Transformer Encoder
        # (batch, seq_len/kernel_size, n_filters) → transformer blocks → (batch, seq_len/kernel_size, n_filters)
        x = self.transformer(x)

        # Layer 5a: Global Max Pooling
        # (batch, seq_len/kernel_size, n_filters) → max over sequence → (batch, n_filters)
        x = torch.max(x, dim=1)[0]

        # Layer 5b: Classification Head
        # (batch, n_filters) → FC → (batch, num_classes)
        logits = self.fc(x)

        return logits
