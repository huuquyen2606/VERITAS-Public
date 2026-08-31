from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from .features import extract_raw_binary_features


class ClassifierLike(Protocol):
    def predict_malicious(self, data: bytes) -> tuple[bool, float]:
        ...


def _ensure_ember():
    try:
        import ember  # type: ignore

        return ember
    except Exception as exc:
        raise RuntimeError(
            "EMBER is required for LightGBM-EMBER surrogate. "
            "Install ember and ensure it is importable in this environment."
        ) from exc


def extract_classifier_features(data: bytes) -> np.ndarray:
    # Paper-faithful state feature family for original GBDT baseline.
    return extract_raw_binary_features(data).astype(np.float32)


def extract_ember_v2_features(data: bytes) -> np.ndarray:
    ember = _ensure_ember()
    # Paper Section IV-B.3: "byte-level data, header, section and import/exports"
    # Version 1 = ~2350 dims (Gym-Malware/EMBER v1 standard PE feature set)
    extractor = ember.PEFeatureExtractor(1, print_feature_warning=False)
    return np.asarray(extractor.feature_vector(data), dtype=np.float32)


import concurrent.futures
import multiprocessing as mp

_ember_pool: concurrent.futures.ProcessPoolExecutor | None = None

def _get_ember_pool():
    global _ember_pool
    if _ember_pool is None:
        _ember_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=1,
            mp_context=mp.get_context("spawn")
        )
    return _ember_pool

def _worker_extract_ember(data: bytes) -> np.ndarray | None:
    try:
        return extract_ember_v2_features(data)
    except Exception:
        return None

def extract_ember_robust(data: bytes) -> np.ndarray | None:
    global _ember_pool
    pool = _get_ember_pool()
    try:
        future = pool.submit(_worker_extract_ember, data)
        return future.result(timeout=15.0)
    except concurrent.futures.TimeoutError:
        print("\n[-] LIEF Timeout: File parsing hung. Restarting worker...")
        for pid in pool._processes.keys():
            import os, signal
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        pool.shutdown(wait=False)
        _ember_pool = None
        return None
    except concurrent.futures.process.BrokenProcessPool:
        print("\n[-] Segfault detected in Feature Extractor: C++ crashed! Restarting worker...")
        _ember_pool = None
        return None
    except Exception as e:
        print(f"\n[-] Feature Extractor exception: {e}")
        return None


@dataclass
class IndependentClassifier:
    model: GradientBoostingClassifier
    threshold: float = 0.5

    def predict_malicious(self, data: bytes) -> tuple[bool, float]:
        x = extract_classifier_features(data).reshape(1, -1)
        prob = float(self.model.predict_proba(x)[0, 1])
        return prob >= self.threshold, prob

    def save(self, path: str | Path) -> None:
        payload = {"model": self.model, "threshold": self.threshold}
        joblib.dump(payload, path)

    @staticmethod
    def load(path: str | Path) -> "IndependentClassifier":
        payload = joblib.load(path)
        return IndependentClassifier(model=payload["model"], threshold=payload.get("threshold", 0.5))


def train_independent_classifier(
    malicious_samples: list[bytes],
    benign_samples: list[bytes],
    seed: int = 1337,
) -> IndependentClassifier:
    if len(malicious_samples) == 0 or len(benign_samples) == 0:
        raise ValueError("Need both malicious and benign samples to train classifier")

    x_m = np.stack([extract_classifier_features(s) for s in malicious_samples], axis=0)
    x_b = np.stack([extract_classifier_features(s) for s in benign_samples], axis=0)

    x = np.concatenate([x_m, x_b], axis=0)
    y = np.concatenate([
        np.ones(len(x_m), dtype=np.int64),
        np.zeros(len(x_b), dtype=np.int64),
    ])

    clf = GradientBoostingClassifier(random_state=seed)
    clf.fit(x, y)
    return IndependentClassifier(model=clf, threshold=0.5)


@dataclass
class LightGBMEmberClassifier:
    model: lgb.Booster
    threshold: float = 0.9159

    @staticmethod
    def load_model(model_path: str | Path, threshold: float = 0.9159) -> "LightGBMEmberClassifier":
        booster = lgb.Booster(model_file=str(model_path))
        return LightGBMEmberClassifier(model=booster, threshold=threshold)

    def predict_malicious(self, data: bytes) -> tuple[bool, float]:
        features = extract_ember_robust(data)
        if features is None:
            # If C++ crashes or hangs, penalize RL agent heavily to avoid these actions
            return True, 1.0
            
        x = features.reshape(1, -1)
        prob = float(self.model.predict(x)[0])
        return prob >= self.threshold, prob


@dataclass
class ScikitLearnEmberClassifier:
    model: GradientBoostingClassifier
    threshold: float = 0.9

    @staticmethod
    def load_model(model_path: str | Path, threshold: float = 0.9) -> "ScikitLearnEmberClassifier":
        model = joblib.load(str(model_path))
        return ScikitLearnEmberClassifier(model=model, threshold=threshold)

    def predict_malicious(self, data: bytes) -> tuple[bool, float]:
        features = extract_ember_robust(data)
        if features is None:
            return True, 1.0
        # The PKL model expects exactly 2350 features.
        # Original EMBER v1 returns 2351, so we slice the first 2350 features.
        x = features[:2350].reshape(1, -1)
        prob = float(self.model.predict_proba(x)[0, 1])
        return prob >= self.threshold, prob


def save_classifier_artifact(classifier: ClassifierLike, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if isinstance(classifier, IndependentClassifier):
        model_path = out / "independent_classifier.joblib"
        classifier.save(model_path)
        meta = {
            "classifier_type": "gbdt_raw513",
            "model_file": model_path.name,
            "threshold": classifier.threshold,
        }
    elif isinstance(classifier, LightGBMEmberClassifier):
        model_path = out / "independent_classifier_lgbm.txt"
        classifier.model.save_model(str(model_path))
        meta = {
            "classifier_type": "lgbm_ember_v2_2381",
            "model_file": model_path.name,
            "threshold": classifier.threshold,
        }
    elif isinstance(classifier, ScikitLearnEmberClassifier):
        model_path = out / "independent_classifier_sklearn_ember.pkl"
        joblib.dump(classifier.model, model_path)
        meta = {
            "classifier_type": "sklearn_ember_v1_2350",
            "model_file": model_path.name,
            "threshold": classifier.threshold,
        }
    else:
        raise TypeError(f"Unsupported classifier type: {type(classifier)!r}")

    (out / "classifier_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_classifier_artifact(out_dir: str | Path) -> ClassifierLike:
    out = Path(out_dir)
    meta_path = out / "classifier_meta.json"

    if not meta_path.exists():
        legacy = out / "independent_classifier.joblib"
        if legacy.exists():
            return IndependentClassifier.load(legacy)
        raise FileNotFoundError(f"Classifier artifact not found in: {out}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ctype = meta.get("classifier_type")
    model_file = meta.get("model_file")
    threshold = float(meta.get("threshold", 0.5))
    model_path = out / str(model_file)

    if ctype == "gbdt_raw513":
        return IndependentClassifier.load(model_path)
    if ctype == "lgbm_ember_v2_2381":
        return LightGBMEmberClassifier.load_model(model_path=model_path, threshold=threshold)
    if ctype == "sklearn_ember_v1_2350":
        return ScikitLearnEmberClassifier.load_model(model_path=model_path, threshold=threshold)

    raise ValueError(f"Unsupported classifier_type in metadata: {ctype}")
