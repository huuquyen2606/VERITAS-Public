"""
GAPGAN - Discriminator Network
=================================
Paper: "Black-box Adversarial Attacks Against Deep Learning Based Malware
        Binaries Detection with GAN" (ECAI 2020)

Reference: Section 3.2 & 3.3 of the paper.
  "the discriminator performs binary classification with convolutional
   layers and fully-connected layers."

  The Discriminator D learns to *distill* (imitate) the target black-box
  detector f.  It is trained with labels obtained by querying f, NOT with
  ground-truth labels.

Architecture:
  Input (batch, 1, total_length)       ← total_length = t + payload_length
    → Conv1d layers (feature extraction)
    → Flatten → FC layers
    → Sigmoid output (probability of being benign)

Loss (Eq. 5):
  L_D = E_{x~x_adv}[ H(D(x), f(x)) ] + E_{x~x_ben}[ H(D(x), f(x)) ]
  where H is binary cross-entropy.
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    GAPGAN Discriminator.

    Takes a full sample (original file + appended payloads for adversarial
    samples, or original benign file) and outputs a scalar probability of
    being classified as benign.

    Parameters
    ----------
    input_length : int
        Total length of the input to the discriminator.
        For adversarial samples: t + payload_length.
        For benign samples: t (padded to same total_length).
    channels : int
        Base number of channels for convolution layers.
    """

    def __init__(self, input_length: int, channels: int = 128):
        super().__init__()
        self.input_length = input_length

        # Paper: "convolutional layers" for feature extraction
        self.features = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=512, stride=512),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels * 2, kernel_size=8, stride=8),
            nn.LeakyReLU(0.2),
        )

        # Adaptive pooling normalizes sequence length to a fixed size,
        # making the discriminator robust to variable input lengths.
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Paper: "fully-connected layers" for classification
        self.classifier = nn.Sequential(
            nn.Linear(channels * 2, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, total_length)
            A normalized continuous sample in [-1, 1].

        Returns
        -------
        prob : Tensor of shape (batch,)
            Probability of being classified as benign by D.
        """
        # Add channel dim: (batch, 1, total_length)
        x = x.unsqueeze(1)

        # Convolutional feature extraction
        x = self.features(x)  # (batch, C*2, L')

        # Adaptive pool → (batch, C*2, 1)
        x = self.pool(x).squeeze(-1)

        # Classify
        prob = self.classifier(x).squeeze(-1)  # (batch,)
        return prob
