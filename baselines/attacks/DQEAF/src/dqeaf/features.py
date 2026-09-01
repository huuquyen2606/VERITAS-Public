from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    entropy_window: int = 2048
    entropy_stride: int = 1024
    entropy_bins: int = 16
    byte_bins: int = 16


def _byte_histogram(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size == 0:
        return np.zeros(256, dtype=np.float32)
    hist = np.bincount(arr, minlength=256).astype(np.float32)
    hist /= hist.sum()
    return hist


def _shannon_entropy(block: np.ndarray) -> float:
    if block.size == 0:
        return 0.0
    counts = np.bincount(block, minlength=256).astype(np.float64)
    probs = counts[counts > 0] / block.size
    return float(-(probs * np.log2(probs)).sum())


def _entropy_histogram(data: bytes, cfg: FeatureConfig) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    hist = np.zeros((cfg.entropy_bins, cfg.byte_bins), dtype=np.float32)
    if arr.size == 0:
        return hist.reshape(-1)

    window = cfg.entropy_window
    stride = cfg.entropy_stride
    if arr.size < window:
        entropy = _shannon_entropy(arr)
        entropy_bin = min(cfg.entropy_bins - 1, int(entropy / 8.0 * cfg.entropy_bins))
        byte_bin = min(cfg.byte_bins - 1, int(arr.mean() / 256.0 * cfg.byte_bins))
        hist[entropy_bin, byte_bin] += 1.0
    else:
        for start in range(0, arr.size - window + 1, stride):
            block = arr[start : start + window]
            entropy = _shannon_entropy(block)
            entropy_bin = min(cfg.entropy_bins - 1, int(entropy / 8.0 * cfg.entropy_bins))
            byte_bin = min(cfg.byte_bins - 1, int(block.mean() / 256.0 * cfg.byte_bins))
            hist[entropy_bin, byte_bin] += 1.0

    total = hist.sum()
    if total > 0:
        hist /= total
    return hist.reshape(-1)


def extract_raw_binary_features(data: bytes, cfg: FeatureConfig | None = None) -> np.ndarray:
    """Return the 513-d observation used by DQEAF (paper Section III-C)."""
    cfg = cfg or FeatureConfig()
    bh = _byte_histogram(data)
    eh = _entropy_histogram(data, cfg)
    # 1 additional scalar so total dim = 256 + 256 + 1 = 513.
    bias_feature = np.array([1.0], dtype=np.float32)

    feat = np.concatenate([bh, eh, bias_feature], axis=0).astype(np.float32)
    # Paper normalizes observations to [-0.5, 0.5] in experiment setup.
    return feat - 0.5
