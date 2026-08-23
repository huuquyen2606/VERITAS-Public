"""State-vector builder for the VERITAS agent."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import zlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import numpy as np


logger = logging.getLogger(__name__)

DIM_STATIC = 2381
DIM_DYN = 184
DIM_CODE = 112
DIM_ENV = 40
STATE_DIM = DIM_STATIC + DIM_DYN + DIM_CODE + DIM_ENV
NUM_ACTIONS = 19
NUM_FAMILIES = 5


@dataclass(frozen=True)
class StateBuildResult:
    state: np.ndarray
    static: np.ndarray
    dynamic: np.ndarray
    code: np.ndarray
    env: np.ndarray
    sandbox_context: Dict[str, Any]
    static_source: str


class AdvStateBuilder:
    """Build VERITAS observations from Adv-RL Env Redis data."""

    def __init__(
        self,
        family_to_index: Optional[Mapping[str, int]] = None,
        max_steps: int = 5,
    ) -> None:
        self.family_to_index = dict(family_to_index or {})
        self.max_steps = max(1, int(max_steps))
        self._ember_extractor: Optional[Any] = None

    def build_from_redis_hash(
        self,
        data: Mapping[str, Any],
        *,
        step_count: int = 0,
        previous_actions: Optional[Sequence[int]] = None,
        true_label: Optional[str] = None,
        pe_bytes: Optional[bytes] = None,
        pe_path: Optional[Union[str, Path]] = None,
    ) -> StateBuildResult:
        api_chain = _json_list(data.get("api_chain", "[]"))
        pe_imports = _json_list(data.get("pe_imports", "[]"))
        pe_sections = _json_list(data.get("pe_sections", "[]"))
        signatures = _json_list(data.get("signatures", "[]"))
        byte_sequence = _json_list(data.get("byte_sequence", "[]"))
        cfg = _json_dict(data.get("cfg", "{}"))
        detector_scores = _json_float_dict(data.get("detector_scores", "{}"))

        static, static_source = self._build_static(
            byte_sequence,
            pe_imports,
            pe_sections,
            signatures,
            pe_bytes=pe_bytes,
            pe_path=pe_path,
        )
        dynamic = self._build_dynamic(api_chain, pe_imports, pe_sections, signatures, data)
        code = self._build_code(cfg)
        env = self._build_env(data, cfg, detector_scores, step_count, previous_actions, true_label)

        state = np.concatenate([static, dynamic, code, env]).astype(np.float32)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"state shape {state.shape} != ({STATE_DIM},)")

        sandbox_context = self.build_sandbox_context(data)
        return StateBuildResult(
            state=state,
            static=static,
            dynamic=dynamic,
            code=code,
            env=env,
            sandbox_context=sandbox_context,
            static_source=static_source,
        )

    def build_sandbox_context(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        api_chain_raw = _json_list(data.get("api_chain", "[]"))
        imports_raw = _json_list(data.get("pe_imports", "[]"))
        api_chain = _normalize_api_context_names(api_chain_raw)
        imports = _normalize_api_context_names(imports_raw)
        trace_count = _to_float(data.get("trace_count", 0.0))
        integrity = _to_float(data.get("integrity_score", 1.0))
        functionality = _to_float(data.get("functionality_score", 100.0))

        ordered_api_sequence = api_chain
        observed_api_set = sorted(set(api_chain))
        imported_api_set = sorted(set(imports))
        target_api_set = sorted(set(observed_api_set) & set(imported_api_set))

        return {
            "ordered_api_sequence": ordered_api_sequence,
            "observed_api_set": observed_api_set,
            "imported_api_set": imported_api_set,
            "target_api_set": target_api_set,
            "sandbox_valid": bool(api_chain or trace_count > 0),
            "trace_count": int(trace_count),
            "integrity_score": integrity,
            "functionality_score": functionality,
        }

    def _build_static(
        self,
        byte_sequence: Sequence[Any],
        pe_imports: Sequence[Any],
        pe_sections: Sequence[Any],
        signatures: Sequence[Any],
        *,
        pe_bytes: Optional[bytes] = None,
        pe_path: Optional[Union[str, Path]] = None,
    ) -> tuple[np.ndarray, str]:
        ember_static = self._build_ember_static(pe_bytes=pe_bytes, pe_path=pe_path)
        if ember_static is not None:
            return ember_static, "ember_v2"

        out = np.zeros(DIM_STATIC, dtype=np.float32)

        raw = np.asarray([_to_float(x) for x in byte_sequence[:1024]], dtype=np.float32)
        if raw.size:
            out[: raw.size] = np.clip(raw, 0.0, 255.0) / 255.0

        import_slice = out[1024:2024]
        _hash_tokens_into(import_slice, _flatten_strings(pe_imports), binary=True)

        section_slice = out[2024:2152]
        numerics = _extract_numerics(pe_sections)
        for idx, value in enumerate(numerics[: section_slice.size]):
            section_slice[idx] = _safe_log_scale(value)

        signature_slice = out[2152:]
        _hash_tokens_into(signature_slice, _flatten_strings(signatures), binary=True)

        return out, "redis_fallback"

    def _build_ember_static(
        self,
        *,
        pe_bytes: Optional[bytes] = None,
        pe_path: Optional[Union[str, Path]] = None,
    ) -> Optional[np.ndarray]:
        bytez = pe_bytes
        if bytez is None and pe_path is not None:
            path = Path(pe_path)
            if path.is_file():
                bytez = path.read_bytes()

        if not bytez:
            return None

        try:
            _patch_ember_numpy_aliases()
            _patch_ember_lief_exception_aliases()
            _patch_ember_feature_hasher()
            from ember.features import PEFeatureExtractor

            if self._ember_extractor is None:
                self._ember_extractor = PEFeatureExtractor(
                    feature_version=2,
                    print_feature_warning=False,
                )
            vec = np.asarray(self._ember_extractor.feature_vector(bytez), dtype=np.float32)
        except Exception as exc:
            logger.debug("EMBER static extraction failed; using Redis fallback: %s", exc)
            return None

        if vec.shape != (DIM_STATIC,):
            logger.debug("EMBER static shape %s != (%d,); using Redis fallback.", vec.shape, DIM_STATIC)
            return None

        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _build_dynamic(
        self,
        api_chain: Sequence[Any],
        pe_imports: Sequence[Any],
        pe_sections: Sequence[Any],
        signatures: Sequence[Any],
        data: Mapping[str, Any],
    ) -> np.ndarray:
        out = np.zeros(DIM_DYN, dtype=np.float32)

        api_tokens = [str(api) for api in api_chain]
        api_hist = out[:128]
        _hash_tokens_into(api_hist, api_tokens, binary=False)
        if api_tokens:
            api_hist /= max(1.0, float(len(api_tokens)))

        dll_or_import_hist = out[128:160]
        _hash_tokens_into(dll_or_import_hist, _flatten_strings(pe_imports), binary=False)
        if pe_imports:
            dll_or_import_hist /= max(1.0, float(len(pe_imports)))

        stats = out[160:184]
        unique_apis = len(set(api_tokens))
        imports = list(_flatten_strings(pe_imports))
        sections = list(pe_sections)
        signatures_flat = list(_flatten_strings(signatures))
        trace_count = _to_float(data.get("trace_count", 0.0))
        detector_scores = _json_dict(data.get("detector_scores", "{}"))

        scalars = [
            len(api_tokens),
            unique_apis,
            len(imports),
            len(set(imports)),
            len(sections),
            len(signatures_flat),
            trace_count,
            _to_float(data.get("is_running", 0.0)),
            _to_float(data.get("integrity_score", 1.0)),
            _to_float(data.get("functionality_score", 100.0)) / 100.0,
            _to_float(data.get("reward", 0.0)) / 500.0,
            max(detector_scores.values(), default=0.0),
            _entropy_like(api_tokens),
            _entropy_like(imports),
        ]
        for idx, value in enumerate(scalars[: stats.size]):
            stats[idx] = _safe_log_scale(value) if idx < 7 else float(value)

        return out

    def _build_code(self, cfg: Mapping[str, Any]) -> np.ndarray:
        out = np.zeros(DIM_CODE, dtype=np.float32)
        nodes = _to_float(cfg.get("nodes", 0.0))
        edges = _to_float(cfg.get("edges", 0.0))
        status = str(cfg.get("status", "")).lower()
        avg_degree = edges / nodes if nodes > 0 else 0.0

        scalars = [
            nodes,
            edges,
            avg_degree,
            1.0 if status == "ok" else 0.0,
            1.0 if status == "timeout" else 0.0,
            1.0 if status in {"error", "exception", "subprocess_error"} else 0.0,
        ]
        for idx, value in enumerate(scalars):
            out[idx] = _safe_log_scale(value) if idx < 3 else float(value)
        return out

    def _build_env(
        self,
        data: Mapping[str, Any],
        cfg: Mapping[str, Any],
        detector_scores: Mapping[str, float],
        step_count: int,
        previous_actions: Optional[Sequence[int]],
        true_label: Optional[str],
    ) -> np.ndarray:
        out = np.zeros(DIM_ENV, dtype=np.float32)

        step = max(0, int(step_count))
        remaining = max(0, self.max_steps - step)
        api_count = len(_json_list(data.get("api_chain", "[]")))
        import_count = len(_json_list(data.get("pe_imports", "[]")))
        section_count = len(_json_list(data.get("pe_sections", "[]")))
        trace_count = _to_float(data.get("trace_count", 0.0))
        nodes = _to_float(cfg.get("nodes", 0.0))
        edges = _to_float(cfg.get("edges", 0.0))
        original_prob = _detector_prob(detector_scores, true_label)
        max_prob = max(detector_scores.values(), default=0.0)
        predicted_label = max(detector_scores.items(), key=lambda item: item[1])[0] if detector_scores else ""

        scalars = [
            step / self.max_steps,
            remaining / self.max_steps,
            _to_float(data.get("integrity_score", 1.0)),
            _to_float(data.get("functionality_score", 100.0)) / 100.0,
            _to_float(data.get("reward", 0.0)) / 500.0,
            1.0 if api_count or trace_count > 0 else 0.0,
            _safe_log_scale(trace_count),
            _safe_log_scale(api_count),
            _safe_log_scale(import_count),
            _safe_log_scale(section_count),
            _safe_log_scale(nodes),
            _safe_log_scale(edges),
            original_prob,
            max_prob,
            1.0 if true_label and predicted_label and predicted_label.lower() != str(true_label).lower() else 0.0,
        ]
        out[: len(scalars)] = np.asarray(scalars, dtype=np.float32)

        family = np.zeros(NUM_FAMILIES, dtype=np.float32)
        family_idx = self._family_index(true_label)
        if family_idx is not None and 0 <= family_idx < NUM_FAMILIES:
            family[family_idx] = 1.0
        out[15:20] = family

        actions = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for action in previous_actions or []:
            action_idx = int(action)
            if 0 <= action_idx < NUM_ACTIONS:
                actions[action_idx] = 1.0
        out[20:20 + NUM_ACTIONS] = actions

        return out

    def _family_index(self, true_label: Optional[str]) -> Optional[int]:
        if true_label is None:
            return None
        if true_label in self.family_to_index:
            return int(self.family_to_index[true_label])
        label_norm = str(true_label).lower()
        for label, idx in self.family_to_index.items():
            if str(label).lower() == label_norm:
                return int(idx)
        return None


def _json_list(raw: Any) -> list[Any]:
    parsed = _json_value(raw, default=[])
    return parsed if isinstance(parsed, list) else []


def _json_dict(raw: Any) -> Dict[str, Any]:
    parsed = _json_value(raw, default={})
    return parsed if isinstance(parsed, dict) else {}


def _json_float_dict(raw: Any) -> Dict[str, float]:
    parsed = _json_dict(raw)
    return {str(key): _to_float(value) for key, value in parsed.items()}


def _json_value(raw: Any, default: Any) -> Any:
    if raw is None:
        return default
    if isinstance(raw, (list, dict)):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return default


def _to_float(value: Any) -> float:
    try:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        return float(value)
    except Exception:
        return 0.0


def _safe_log_scale(value: Any) -> float:
    value_f = max(0.0, _to_float(value))
    return float(math.log1p(value_f) / 10.0)


def _stable_hash(token: str, size: int) -> int:
    return zlib.crc32(token.encode("utf-8", errors="ignore")) % size


def _hash_tokens_into(target: np.ndarray, tokens: Iterable[str], binary: bool) -> None:
    size = int(target.size)
    if size <= 0:
        return
    for token in tokens:
        idx = _stable_hash(str(token), size)
        if binary:
            target[idx] = 1.0
        else:
            target[idx] += 1.0


def _flatten_strings(obj: Any) -> Iterable[str]:
    if obj is None:
        return
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            yield str(key)
            yield from _flatten_strings(value)
    elif isinstance(obj, (list, tuple, set)):
        for value in obj:
            yield from _flatten_strings(value)
    elif isinstance(obj, str):
        yield obj
    else:
        yield str(obj)


def _extract_numerics(obj: Any) -> list[float]:
    values: list[float] = []
    if isinstance(obj, Mapping):
        for value in obj.values():
            values.extend(_extract_numerics(value))
    elif isinstance(obj, (list, tuple, set)):
        for value in obj:
            values.extend(_extract_numerics(value))
    else:
        try:
            values.append(float(obj))
        except Exception:
            pass
    return values


def _entropy_like(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    counts: Dict[str, int] = {}
    for token in tokens:
        counts[str(token)] = counts.get(str(token), 0) + 1
    total = float(len(tokens))
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy / 10.0


def _detector_prob(detector_scores: Mapping[str, float], true_label: Optional[str]) -> float:
    if not detector_scores or true_label is None:
        return 1.0
    if true_label in detector_scores:
        return float(detector_scores[true_label])
    label_norm = str(true_label).lower()
    for label, value in detector_scores.items():
        if str(label).lower() == label_norm:
            return float(value)
    return 1.0


def _normalize_api_context_names(values: Sequence[Any]) -> list[str]:
    names: list[str] = []
    for value in values:
        name = _api_context_name(value)
        if name:
            names.append(name)
    return names


def _api_context_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in ("api", "api_name", "name", "function", "call"):
            if key in value:
                return _api_context_name(value.get(key))
        return None
    if isinstance(value, (tuple, list)):
        if not value:
            return None
        return _api_context_name(value[0])
    text = value.decode("utf-8", errors="ignore") if isinstance(value, bytes) else str(value)
    text = text.strip().strip("\x00")
    if "!" in text:
        text = text.rsplit("!", 1)[1]
    if not text or text in {"_PAD_", "__exception__", "__anomaly__"}:
        return None
    return text


def _patch_ember_lief_exception_aliases() -> None:
    """Alias missing LIEF legacy exceptions to RuntimeError for EMBER compatibility."""
    try:
        import lief
    except Exception:
        return

    for name in (
        "bad_format",
        "bad_file",
        "pe_error",
        "parser_error",
        "read_out_of_bound",
    ):
        if not hasattr(lief, name):
            setattr(lief, name, RuntimeError)


def _patch_ember_numpy_aliases() -> None:
    """Provide NumPy aliases still referenced by the public EMBER package."""
    if not hasattr(np, "int"):
        setattr(np, "int", int)


def _patch_ember_feature_hasher() -> None:
    """Adapt EMBER's old FeatureHasher calls to current scikit-learn."""
    try:
        from sklearn.feature_extraction import FeatureHasher
    except Exception:
        return

    original_transform = FeatureHasher.transform
    if getattr(original_transform, "_veritas_ember_compat", False):
        return

    def transform(self: Any, raw_X: Any) -> Any:
        if getattr(self, "input_type", None) == "string":
            if isinstance(raw_X, list) and raw_X and all(isinstance(x, str) for x in raw_X):
                raw_X = [[x] for x in raw_X]
        return original_transform(self, raw_X)

    setattr(transform, "_veritas_ember_compat", True)
    FeatureHasher.transform = transform
