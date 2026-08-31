"""
GAPGAN - Feature Extraction / Preprocessor
============================================
Paper: "Black-box Adversarial Attacks Against Deep Learning Based Malware
        Binaries Detection with GAN" (ECAI 2020)

Handles:
  1. Reading raw PE binaries as byte arrays.
  2. Zero-padding each binary to a fixed length t.
  3. Normalizing byte values from {0,...,255} to [-1, 1].
  4. Inverse mapping from continuous [-1, 1] back to discrete {0,...,255}.

Reference: Section 3.1 & 3.2 of the paper.
  - b = (b1, ..., bn) ∈ {0,...,255}^n
  - b0 = (b1, ..., bn, 0, ..., 0) ∈ {0,...,255}^t   (zero-padding)
  - x  = normalize(b0) ∈ [-1, 1]^t
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Core transformation functions
# ---------------------------------------------------------------------------

def read_binary(filepath: str) -> np.ndarray:
    """Read a PE binary file and return it as a uint8 numpy array."""
    with open(filepath, "rb") as f:
        raw = f.read()
    return np.frombuffer(raw, dtype=np.uint8)


def pad_binary(byte_array: np.ndarray, target_length: int) -> np.ndarray:
    """
    Zero-pad (or truncate) a byte array to *target_length*.

    Paper §3.2:
        "we first append zeros to the end of input binaries to match
         the input size t of the network"
    """
    n = len(byte_array)
    if n >= target_length:
        return byte_array[:target_length]
    padded = np.zeros(target_length, dtype=np.uint8)
    padded[:n] = byte_array
    return padded


def normalize(byte_array: np.ndarray) -> np.ndarray:
    """
    Map byte values {0,...,255} → [-1, 1].

    Paper §3.2:
        "we map each byte in discrete binaries to a continuous space
         [-1, 1] by normalization"
    """
    return byte_array.astype(np.float32) / 127.5 - 1.0


def denormalize(continuous_array: np.ndarray) -> np.ndarray:
    """
    Inverse of *normalize*: map [-1, 1] → {0,...,255}.
    Clips to valid byte range before casting.
    """
    out = (continuous_array + 1.0) * 127.5
    out = np.clip(np.round(out), 0, 255).astype(np.uint8)
    return out


def normalized_to_byte_input(x_normalized: torch.Tensor) -> torch.Tensor:
    """
    Convert normalized [-1,1] tensor back to integer byte values {0..255}
    for querying the MalConv black-box detector.

    Parameters
    ----------
    x_normalized : Tensor of shape (batch, length)
        Samples in continuous [-1, 1] space.

    Returns
    -------
    byte_vals : LongTensor of shape (batch, length)
        Integer byte values in {0, ..., 255}.
    """
    byte_vals = ((x_normalized + 1.0) * 127.5).round().clamp(0, 255).long()
    return byte_vals


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class BinaryDataset(Dataset):
    """
    PyTorch Dataset that loads raw PE binaries from *malware_dir* and
    *benign_dir*, pads them to *target_length*, and normalizes to [-1, 1].

    Labels follow the paper convention:
        malware  → y = -1
        benign   → y =  1
    """

    def __init__(self, malware_dir: str, benign_dir: str,
                 target_length: int = 2_000_000):
        super().__init__()
        self.target_length = target_length
        self.samples = []  # list of (filepath, label)

        all_mal_files = self._list_files(malware_dir)
        self.benign_files = self._list_files(benign_dir)
        
        # Ensure malware and benign sets are disjoint if directories overlap
        benign_set = set(self.benign_files)
        self.mal_files = [f for f in all_mal_files if f not in benign_set]

        for fpath in self.mal_files:
            self.samples.append((fpath, -1))

        for fpath in self.benign_files:
            self.samples.append((fpath, 1))

    def _list_files(self, directory: str) -> list:
        """List all files in directory recursively."""
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                if os.path.isfile(file_path):
                    files.append(file_path)
        return sorted(files)  # Sort for reproducibility

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        raw = read_binary(filepath)
        original_length = len(raw)
        padded = pad_binary(raw, self.target_length)
        normalized = normalize(padded)
        x = torch.from_numpy(normalized)           # shape: (t,)
        y = torch.tensor(label, dtype=torch.long)
        return x, y, original_length
