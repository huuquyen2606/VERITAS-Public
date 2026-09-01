import torch
import torch.nn as nn


class MultiViewCNNArch(nn.Module):
    """
    1D Convolutional Neural Network for Multi-View Malware Classification.

    Paper: "A multi-view feature fusion approach for effective malware
            classification using Deep Learning" (Chaganti et al., 2023)

    Expected Input Shape: (batch_size, 1, 2128)
    Output Shape: (batch_size, 1) - Binary Classification
    """

    def __init__(self):
        super(MultiViewCNNArch, self).__init__()

        # -------------------------------------------------------------------
        # 1. Feature Extractor (1D Convolution)
        # Matches Section 3.2 and Figure 1 of the paper exactly
        # -------------------------------------------------------------------
        self.features = nn.Sequential(
            # Input: (batch_size, 1, 2128) -> Output: (batch_size, 64, 2128)
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Max Pooling -> Output: (batch_size, 64, 1064)
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # -------------------------------------------------------------------
        # 2. Classifier (Dense Layers)
        # Matches Section 3.2: 128 units, 0.5 Dropout, Sigmoid 1-unit output
        # -------------------------------------------------------------------
        # Flattened size: 64 channels * 1064 sequence length = 68096
        self.flatten_size = 64 * 1064

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # First Dense Layer: 128 units with ReLU and Dropout
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            # Final Classification Layer: 1 unit for Binary Classification
            # Note: We output raw logits here. The Sigmoid activation is applied
            # by BCEWithLogitsLoss during training and manually during prediction.
            nn.Linear(128, 1),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape (batch_size, 1, 2128)

        Returns:
            Tensor: Raw logit of shape (batch_size, 1)
        """
        # Extract features through Conv1D layer
        x = self.features(x)

        # Flatten and classify
        logits = self.classifier(x)

        return logits
