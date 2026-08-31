import os
import site
import sys
import importlib.util
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from obfumal.utils.lief_compat import ensure_legacy_exception_aliases


def _patch_numpy_compat():
    # ember uses deprecated numpy aliases on newer numpy versions.
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "bool"):
        np.bool = bool  # type: ignore[attr-defined]


def _augment_import_paths():
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        site.getusersitepackages(),
        f"/usr/lib/python{pyver}/dist-packages",
        f"/usr/lib/python{sys.version_info.major}/dist-packages",
        "/usr/lib/python3/dist-packages",
        f"/usr/local/lib/python{pyver}/dist-packages",
        f"/usr/local/lib/python{sys.version_info.major}/dist-packages",
        "/usr/local/lib/python3/dist-packages",
    ]
    for path in candidates:
        if path and os.path.exists(path) and path not in sys.path:
            sys.path.append(path)


def _load_ember() -> Tuple[Optional[type], Optional[str]]:
    try:
        _augment_import_paths()
        _patch_numpy_compat()
        ensure_legacy_exception_aliases()
        import ember

        return ember.PEFeatureExtractor, None
    except Exception as e:
        return None, repr(e)


def _load_gym_malware() -> Tuple[Optional[type], Optional[str]]:
    try:
        from gym_malware.envs.utils.pefeatures import PEFeatureExtractor

        return PEFeatureExtractor, None
    except Exception as e:
        return None, repr(e)


def _load_gym_malware_local() -> Tuple[Optional[type], Optional[str]]:
    try:
        repo_root = Path(__file__).resolve().parents[3]
        gm_path = repo_root / "gym-malware"
        pefeatures_path = gm_path / "gym_malware" / "envs" / "utils" / "pefeatures.py"
        if pefeatures_path.exists():
            spec = importlib.util.spec_from_file_location("obfumal._pefeatures", str(pefeatures_path))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "PEFeatureExtractor"):
                    return module.PEFeatureExtractor, None
        return None, "gym-malware local extractor file not found"
    except Exception as e:
        return None, repr(e)


def _import_pefeatures(preferred_backend: str = "auto") -> Tuple[Optional[type], Optional[str], Dict[str, str]]:
    loaders = {
        "ember": _load_ember,
        "gym-malware": _load_gym_malware,
        "gym-malware-local": _load_gym_malware_local,
    }
    errors: Dict[str, str] = {}

    if preferred_backend != "auto":
        if preferred_backend not in loaders:
            raise RuntimeError(f"Unsupported feature backend: {preferred_backend}")
        cls, err = loaders[preferred_backend]()
        if cls is not None:
            return cls, preferred_backend, errors
        errors[preferred_backend] = err or "unknown error"
        return None, None, errors

    for backend in ("ember", "gym-malware", "gym-malware-local"):
        cls, err = loaders[backend]()
        if cls is not None:
            return cls, backend, errors
        errors[backend] = err or "unknown error"
    return None, None, errors


class FeatureExtractor:
    def __init__(self, feature_dimension: int = 2381, backend: str = "auto"):
        self.feature_dimension = feature_dimension
        extractor_cls, source, errors = _import_pefeatures(preferred_backend=backend)
        if extractor_cls is None:
            details = "; ".join(f"{k}: {v}" for k, v in errors.items())
            raise RuntimeError(
                "No PE feature extractor found. Install gym-malware or ember to extract 2381-dim features. "
                f"Backend errors: {details}"
            )

        self._source = source
        if source == "ember":
            self._extractor = extractor_cls(2, print_feature_warning=False)
        else:
            self._extractor = extractor_cls()

    @property
    def source(self) -> Optional[str]:
        return self._source

    def extract(self, bytez: bytes) -> np.ndarray:
        if hasattr(self._extractor, "feature_vector"):
            features = self._extractor.feature_vector(bytez)
        else:
            features = self._extractor.extract(bytez)

        features = np.asarray(features, dtype=np.float32).flatten()
        if features.shape[0] != self.feature_dimension:
            raise ValueError(
                f"Feature dimension mismatch: got {features.shape[0]}, "
                f"expected {self.feature_dimension}. "
                "Check detector model and feature_backend compatibility."
            )
        return features
