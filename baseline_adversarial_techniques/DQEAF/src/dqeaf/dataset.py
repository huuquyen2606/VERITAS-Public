from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def load_binary_corpus(
    root: str | Path,
    max_files: int | None = None,
    exclude_dir_names: Sequence[str] | None = None,
) -> list[bytes]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")

    excluded = {x.strip().lower() for x in (exclude_dir_names or []) if x.strip()}
    files: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if excluded:
            rel_parts = p.relative_to(root).parts[:-1]
            if any(part.lower() in excluded for part in rel_parts):
                continue
        files.append(p)
    files.sort()

    if max_files is not None:
        files = files[:max_files]

    corpus: list[bytes] = []
    for p in files:
        try:
            corpus.append(p.read_bytes())
        except Exception:
            continue
    return corpus


def train_test_split_bytes(
    data: Sequence[bytes],
    test_size: float = 0.2,
    seed: int = 1337,
) -> tuple[list[bytes], list[bytes]]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")
    n = len(data)
    if n < 2:
        raise ValueError("Need at least two samples for split")

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(n * (1.0 - test_size))
    train_idx = idx[:cut]
    test_idx = idx[cut:]

    train = [data[i] for i in train_idx]
    test = [data[i] for i in test_idx]
    return train, test
