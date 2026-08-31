from typing import Any, Dict, Optional

from obfumal.detectors.base import Detector
from obfumal.env.state import FeatureExtractor


class LightGBMDetector(Detector):
    def __init__(
        self,
        model_path: str,
        threshold: float = 0.9,
        feature_extractor: Optional[FeatureExtractor] = None,
        feature_backend: str = "auto",
    ):
        import lightgbm as lgb

        self.model_path = model_path
        self.threshold = threshold
        self.feature_backend = feature_backend
        self.feature_extractor = feature_extractor or FeatureExtractor(backend=feature_backend)
        self.model = lgb.Booster(model_file=model_path)

    @property
    def name(self) -> str:
        return "LightGBM"

    def predict(self, bytez: bytes) -> Dict[str, Any]:
        features = self.feature_extractor.extract(bytez)
        score = float(self.model.predict(features.reshape(1, -1))[0])
        malicious = score >= self.threshold
        return {"malicious": malicious, "label": 1 if malicious else 0, "score": score}


class ThresholdWrapper(Detector):
    def __init__(self, base: Detector, threshold: float = 0.5):
        self.base = base
        self.threshold = threshold

    @property
    def name(self) -> str:
        return f"Threshold({self.base.name})"

    def predict(self, bytez: bytes) -> Dict[str, Any]:
        pred = self.base.predict(bytez)
        score = float(pred.get("score", 1.0))
        malicious = score >= self.threshold
        pred["malicious"] = malicious
        pred["label"] = 1 if malicious else 0
        return pred
