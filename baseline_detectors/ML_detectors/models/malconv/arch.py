import torch
import torch.nn as nn
import torch.nn.functional as F


class MalConv(nn.Module):
    """
    MalConv architecture for multi-class malware classification.
    Refined based on (Al Kadri et al., 2019).

    Architecture:
    - Embedding layer (8 dimensions)
    - Two parallel 1D convolutions (128 filters, 500 kernel, 500 stride)
    - Gating mechanism (element-wise multiplication + Sigmoid + ReLU)
    - Global max pooling
    - Fully connected classifier
    """

    def __init__(
        self,
        vocab_size=257,
        emb_dim=8,
        n_filters=128,
        kernel_size=500,
        stride=500,
        num_classes=6,
    ):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=256)

        # Two parallel convolutional layers
        self.conv_a = nn.Conv1d(
            in_channels=emb_dim,
            out_channels=n_filters,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.conv_b = nn.Conv1d(
            in_channels=emb_dim,
            out_channels=n_filters,
            kernel_size=kernel_size,
            stride=stride,
        )

        # Fully connected layer to output classes
        self.fc = nn.Linear(n_filters, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch_size, max_len)

        Returns:
            tuple: (logits, pooled_features)
                - logits: (batch_size, num_classes) for classification
                - pooled_features: (batch_size, n_filters) for DeCov regularization
        """
        # Layer 1: Embedding
        emb = self.embedding(x)
        # Permute for Conv1d: (batch_size, max_len, emb_dim) → (batch_size, emb_dim, max_len)
        emb = emb.permute(0, 2, 1)

        # Layer 2: Convolution
        a = self.conv_a(emb)
        b = self.conv_b(emb)

        # Layer 3: Gating mechanism with ReLU
        # Paper: "optionally passed to a rectified linear unit (ReLU)"
        gated = F.relu(a * torch.sigmoid(b))

        # Layer 4: Global max pooling
        # (batch_size, n_filters)
        pooled = torch.max(gated, dim=2)[0]

        # Layer 5: Fully connected classifier
        logits = self.fc(pooled)

        # Trả về cả 2 để file train xử lý DeCov Loss
        return logits, pooled
