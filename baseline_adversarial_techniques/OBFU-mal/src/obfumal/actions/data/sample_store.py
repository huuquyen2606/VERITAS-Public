import os
import re
from typing import Dict, List, Optional, Sequence

from obfumal.utils.hashing import sha256_bytes


class SampleStore:
    def __init__(self, sample_dir: str, sha256_list: Optional[Sequence[str]] = None, cache: bool = False):
        self.sample_dir = sample_dir
        self.cache = cache
        self._cache: Dict[str, bytes] = {}
        self._index: Dict[str, str] = {}

        if sha256_list is not None:
            for sha in sha256_list:
                self._index[sha] = os.path.join(sample_dir, sha)
        else:
            self._index_samples()

    def _index_samples(self):
        if not os.path.exists(self.sample_dir):
            return

        for fname in os.listdir(self.sample_dir):
            path = os.path.join(self.sample_dir, fname)
            if not os.path.isfile(path):
                continue
            if re.fullmatch(r"[0-9a-fA-F]{64}", fname):
                self._index[fname] = path

    def list(self) -> List[str]:
        return list(self._index.keys())

    def fetch(self, sha256: str) -> bytes:
        if self.cache and sha256 in self._cache:
            return self._cache[sha256]

        if sha256 not in self._index:
            raise FileNotFoundError(f"Sample {sha256} not found in {self.sample_dir}")

        with open(self._index[sha256], "rb") as f:
            bytez = f.read()

        if self.cache:
            self._cache[sha256] = bytez
        return bytez

    def add(self, path: str) -> str:
        with open(path, "rb") as f:
            bytez = f.read()
        sha = sha256_bytes(bytez)
        self._index[sha] = path
        if self.cache:
            self._cache[sha] = bytez
        return sha
