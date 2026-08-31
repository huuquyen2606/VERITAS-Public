import random
from typing import Any, Dict, Sequence


def detector_details(detector: Any) -> Dict[str, Any]:
    feature_extractor = getattr(detector, "feature_extractor", None)
    return {
        "name": getattr(detector, "name", detector.__class__.__name__),
        "model_path": getattr(detector, "model_path", None),
        "threshold": getattr(detector, "threshold", None),
        "feature_source": getattr(feature_extractor, "source", None),
    }


def probe_detector_on_samples(
    detector: Any,
    sample_store: Any,
    sha256_list: Sequence[str],
    sample_size: int = 64,
    seed: int = 123,
) -> Dict[str, Any]:
    n_total = len(sha256_list)
    if n_total == 0:
        return {
            "sampled": 0,
            "malicious_count": 0,
            "score_min": None,
            "score_max": None,
            "score_mean": None,
            "errors": 0,
        }

    n_pick = min(max(sample_size, 1), n_total)
    rng = random.Random(seed)
    picked = list(sha256_list) if n_pick == n_total else rng.sample(list(sha256_list), n_pick)

    scores = []
    malicious_count = 0
    errors = 0
    for sha in picked:
        try:
            bytez = sample_store.fetch(sha)
            pred = detector.predict(bytez)
            score = float(pred.get("score", 0.0))
            scores.append(score)
            if bool(pred.get("malicious", False)):
                malicious_count += 1
        except Exception:
            errors += 1

    if scores:
        score_min = min(scores)
        score_max = max(scores)
        score_mean = sum(scores) / len(scores)
    else:
        score_min = None
        score_max = None
        score_mean = None

    return {
        "sampled": len(picked),
        "malicious_count": malicious_count,
        "score_min": score_min,
        "score_max": score_max,
        "score_mean": score_mean,
        "errors": errors,
    }
