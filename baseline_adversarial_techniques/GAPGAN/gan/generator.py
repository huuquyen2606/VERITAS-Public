"""
GAPGAN - Generator Network
============================
Paper: "Black-box Adversarial Attacks Against Deep Learning Based Malware
        Binaries Detection with GAN" (ECAI 2020)

Reference: Section 3.2 of the paper.
  "the generator first extracts features of inputs with two convolution
   layers. Then, it resizes the high-level features with fully-connected
   layers. After two layers of deconvolution and one layer of 1*1
   convolution, the adversarial payloads are generated."

Architecture:
  Input x_mal (normalized malware, shape: (batch, 1, t))
    → Conv1d  (feature extraction layer 1)
    → Conv1d  (feature extraction layer 2)
    → AdaptiveAvgPool1d(1)
    → FC layers  (resize high-level features to match payload dims)
    → Reshape
    → ConvTranspose1d  (deconvolution layer 1)
    → ConvTranspose1d  (deconvolution layer 2)
    → Conv1d kernel=1  (1×1 convolution, output projection)
    → Tanh              (constrain output to [-1, 1])

Output: adversarial payloads a_adv of length payload_length.
"""

import math

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    GAPGAN Generator.

    Takes a normalized malware sample x_mal ∈ [-1,1]^t and produces
    adversarial payloads a_adv ∈ [-1,1]^payload_length.

    Parameters
    ----------
    input_length : int
        Fixed input length t of the normalized malware sample.
    payload_length : int
        Length of the adversarial payloads to generate.
    channels : int
        Base number of channels for convolution layers.
    """

    def __init__(self, input_length: int, payload_length: int,
                 channels: int = 128):
        super().__init__()
        self.input_length = input_length
        self.payload_length = payload_length
        self.channels = channels

        # --- Feature extraction: two Conv1d layers ---
        # Paper: "extracts features of inputs with two convolution layers"
        self.conv1 = nn.Conv1d(1, channels, kernel_size=512, stride=512)
        self.conv2 = nn.Conv1d(channels, channels * 2, kernel_size=8, stride=8)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # --- Fully-connected resize layer ---
        # Paper only fixes the topology; adaptive pooling keeps the
        # FC stage resource-feasible at input_length=2,000,000.
        self.seed_len = max(1, math.ceil(payload_length / 64))
        self.fc1 = nn.Linear(channels * 2, channels * 2)
        self.fc2 = nn.Linear(channels * 2, (channels * 2) * self.seed_len)

        # --- Deconvolution (transposed convolution) layers ---
        # Paper: "After two layers of deconvolution"
        self.deconv1 = nn.ConvTranspose1d(
            channels * 2, channels,
            kernel_size=8, stride=8
        )
        self.deconv2 = nn.ConvTranspose1d(
            channels, channels // 2,
            kernel_size=8, stride=8
        )

        # --- 1×1 convolution output layer ---
        # Paper: "and one layer of 1*1 convolution, the adversarial
        #         payloads are generated"
        self.conv_out = nn.Conv1d(channels // 2, 1, kernel_size=1)

        # Activation
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, input_length)
            Normalized malware sample x_mal ∈ [-1, 1]^t.

        Returns
        -------
        a_adv : Tensor of shape (batch, payload_length)
            Adversarial payloads in [-1, 1].
        """
        batch_size = x.size(0)

        # Add channel dim: (batch, 1, t)
        x = x.unsqueeze(1)

        # Feature extraction
        x = self.leaky_relu(self.conv1(x))   # (batch, C, L1)
        x = self.leaky_relu(self.conv2(x))   # (batch, 2C, L2)
        x = self.pool(x).squeeze(-1)         # (batch, 2C)

        # FC resize
        x = self.leaky_relu(self.fc1(x))     # (batch, 2C)
        x = self.leaky_relu(self.fc2(x))     # (batch, 2C * seed_len)

        # Reshape for deconvolution
        x = x.view(batch_size, self.channels * 2, self.seed_len)

        # Deconvolution layers
        x = self.leaky_relu(self.deconv1(x)) # (batch, C, seed_len*8)
        x = self.leaky_relu(self.deconv2(x)) # (batch, C//2, seed_len*64)

        # 1×1 conv → squeeze channel dim
        x = self.conv_out(x)                 # (batch, 1, seed_len*64)
        x = x.squeeze(1)                     # (batch, seed_len*64)

        # Trim or pad to exact payload_length
        if x.size(1) >= self.payload_length:
            x = x[:, :self.payload_length]
        else:
            pad = torch.zeros(batch_size,
                              self.payload_length - x.size(1),
                              device=x.device)
            x = torch.cat([x, pad], dim=1)

        # Output activation: constrain to [-1, 1]
        a_adv = self.tanh(x)
        return a_adv
