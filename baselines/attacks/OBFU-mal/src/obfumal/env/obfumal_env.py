import os
import random
import multiprocessing as mp
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

from obfumal.actions.base import Action
from obfumal.actions.classic.overlay_append import OverlayAppend
from obfumal.actions.classic.imports_append import ImportsAppend
from obfumal.actions.classic.section_rename import SectionRename
from obfumal.actions.classic.remove_signature import RemoveSignature
from obfumal.actions.classic.remove_debug import RemoveDebug
from obfumal.actions.classic.section_append import SectionAppend
from obfumal.actions.classic.break_checksum import BreakChecksum
from obfumal.actions.classic.change_timestamp import ChangeTimestamp
from obfumal.actions.obfuscation.pack_adapter import UPXPack
from obfumal.actions.obfuscation.xor_adapter import DarkarmourXOR
from obfumal.actions.data.sample_store import SampleStore
from obfumal.detectors.base import Detector
from obfumal.env.state import FeatureExtractor
from obfumal.reward.sparse import SparseReward
from obfumal.reward.shaped import ShapedReward
from obfumal.utils.hashing import sha256_bytes

try:
    import pefile
except Exception:
    pefile = None

_NATIVE_RISK_ACTIONS = frozenset(
    {
        "ImportsAppend",
        "SectionRename",
        "RemoveSignature",
        "RemoveDebug",
        "SectionAppend",
        "BreakChecksum",
        "ChangeTimestamp",
    }
)


def _is_reasonable_pe(bytez: bytes) -> bool:
    # Fast sanity filter to avoid feeding obviously broken binaries back into LIEF.
    if len(bytez) < 0x100 or bytez[:2] != b"MZ":
        return False
    if pefile is None:
        return True
    try:
        pe = pefile.PE(data=bytez, fast_load=True)
        pe.parse_data_directories(directories=[])
        pe.close()
        return True
    except Exception:
        return False


def _apply_action_worker(action_obj: Action, bytez: bytes, out_conn: "mp.connection.Connection") -> None:
    try:
        result = action_obj.apply(bytez)
        if isinstance(result, bytearray):
            result = bytes(result)
        if not isinstance(result, bytes):
            result = bytez
        out_conn.send_bytes(result)
    except Exception:
        try:
            out_conn.send_bytes(bytez)
        except Exception:
            pass
    finally:
        out_conn.close()


def default_action_set(
    include_obfuscation: bool = True,
    upx_path: str = "upx",
    darkarmour_path: str = "darkarmour",
) -> List[Action]:
    actions: List[Action] = [
        OverlayAppend(),
        ImportsAppend(),
        SectionRename(),
        RemoveSignature(),
        RemoveDebug(),
        SectionAppend(),
        BreakChecksum(),
        ChangeTimestamp(),
        UPXPack(upx_path=upx_path),
    ]
    if include_obfuscation:
        actions.extend(
            [
                DarkarmourXOR(loop_level=1, darkarmour_path=darkarmour_path),
                DarkarmourXOR(loop_level=2, darkarmour_path=darkarmour_path),
                DarkarmourXOR(loop_level=3, darkarmour_path=darkarmour_path),
            ]
        )
    return actions


class BaseMalwareEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        detector: Detector,
        sample_dir: str,
        sha256_list: Optional[Sequence[str]] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        max_turns: int = 5,
        random_sample: bool = True,
        cache_samples: bool = False,
        cache_predictions: bool = True,
        skip_benign: bool = True,
        max_reset_attempts: Optional[int] = None,
        validate_actions: bool = True,
        isolate_native_actions: bool = True,
        action_apply_timeout_sec: float = 2.0,
        output_dir: str = "artifacts/evaded/blackbox",
        action_set: Optional[List[Action]] = None,
        reward_fn: Optional[object] = None,
    ):
        super().__init__()

        self.detector = detector
        self.max_turns = max_turns
        self.random_sample = random_sample
        self.cache_predictions = cache_predictions
        self.skip_benign = skip_benign
        self.max_reset_attempts = max_reset_attempts
        self.validate_actions = validate_actions
        self.isolate_native_actions = isolate_native_actions
        self.action_apply_timeout_sec = float(action_apply_timeout_sec)

        self.feature_extractor = feature_extractor or FeatureExtractor()

        self.action_list = action_set or default_action_set()
        self.action_space = gym.spaces.Discrete(len(self.action_list))

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.feature_extractor.feature_dimension,),
            dtype=np.float32,
        )

        self.sample_store = SampleStore(sample_dir, sha256_list=sha256_list, cache=cache_samples)
        self.available_sha256 = self.sample_store.list()
        if not self.available_sha256:
            raise RuntimeError(f"No samples found in {sample_dir}")
        if self.max_reset_attempts is None:
            # Avoid infinite loops when skip_benign is enabled and detector marks all files benign.
            self.max_reset_attempts = max(1000, len(self.available_sha256) * 2)

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.reward_fn = reward_fn or SparseReward()
        self.history: Dict[str, Dict[str, Any]] = OrderedDict()
        self.episode_history: List[Dict[str, Any]] = []
        self.run_stats: Dict[str, int] = {"episodes_started": 0, "skipped_benign": 0}
        self._current_episode_record: Optional[Dict[str, Any]] = None
        self._pred_cache: Dict[str, Dict[str, Any]] = {}
        self._sample_index = 0

        self.bytez: bytes = b""
        self.sha256: str = ""
        self.original_score: float = 0.0
        self.current_step = 0

    def _apply_action(self, action_obj: Action, bytez: bytes) -> bytes:
        if not self.isolate_native_actions or action_obj.name not in _NATIVE_RISK_ACTIONS:
            try:
                return action_obj.apply(bytez)
            except Exception:
                return bytez

        methods = mp.get_all_start_methods()
        ctx = mp.get_context("fork") if "fork" in methods else mp.get_context()
        recv_conn, send_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_apply_action_worker, args=(action_obj, bytez, send_conn), daemon=True)
        proc.start()
        send_conn.close()
        proc.join(timeout=self.action_apply_timeout_sec)

        if proc.is_alive():
            proc.terminate()
            proc.join()
            recv_conn.close()
            return bytez
        if proc.exitcode != 0:
            recv_conn.close()
            return bytez

        if not recv_conn.poll():
            recv_conn.close()
            return bytez
        try:
            result = recv_conn.recv_bytes()
        except Exception:
            result = bytez
        recv_conn.close()
        return result if isinstance(result, (bytes, bytearray)) else bytez

    def _select_sha256(self) -> str:
        if self.random_sample:
            return random.choice(self.available_sha256)
        sha = self.available_sha256[self._sample_index % len(self.available_sha256)]
        self._sample_index += 1
        return sha

    def _predict(self, bytez: bytes) -> Dict[str, Any]:
        sha = sha256_bytes(bytez)
        if self.cache_predictions and sha in self._pred_cache:
            return self._pred_cache[sha]
        pred = self.detector.predict(bytez)
        if self.cache_predictions:
            self._pred_cache[sha] = pred
        return pred

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        force_sha = options is not None and "sha256" in options
        attempts = 0
        skipped_benign = 0
        last_score = None
        self._current_episode_record = None
        while True:
            attempts += 1
            if force_sha:
                self.sha256 = options["sha256"]
            else:
                self.sha256 = self._select_sha256()

            self.bytez = self.sample_store.fetch(self.sha256)
            pred = self._predict(self.bytez)
            last_score = float(pred.get("score", 0.0))

            if self.skip_benign and not pred.get("malicious", True) and not force_sha:
                skipped_benign += 1
                self.run_stats["skipped_benign"] += 1
                if attempts >= int(self.max_reset_attempts):
                    detector_threshold = getattr(self.detector, "threshold", None)
                    detector_name = getattr(self.detector, "name", self.detector.__class__.__name__)
                    raise RuntimeError(
                        "Environment reset exceeded max attempts while skip_benign=True. "
                        f"attempts={attempts}, skipped={skipped_benign}, last_score={last_score:.6f}, "
                        f"detector={detector_name}, threshold={detector_threshold}. "
                        "Likely causes: threshold too high for this corpus, or feature-extractor/model mismatch."
                    )
                continue

            self.original_score = float(pred.get("score", 1.0))
            record = {
                "original_sha256": self.sha256,
                "actions": [],
                "evaded": False,
                "evaded_sha256": None,
                "original_score": self.original_score,
            }
            self.history[self.sha256] = record
            self.episode_history.append(record)
            self._current_episode_record = record
            self.run_stats["episodes_started"] += 1
            break

        self.current_step = 0
        observation = self.feature_extractor.extract(self.bytez)
        info = {
            "original_score": self.original_score,
            "original_label": pred.get("label", 1),
            "sha256": self.sha256,
        }
        return observation, info

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self._current_episode_record is None:
            raise RuntimeError("step() called before reset() initialized an episode record")

        action_obj = self.action_list[action_idx]
        self._current_episode_record["actions"].append(action_obj.name)

        original_bytez = self.bytez
        if self.validate_actions and not _is_reasonable_pe(original_bytez):
            self.bytez = original_bytez
        else:
            candidate = self._apply_action(action_obj, original_bytez)
            if self.validate_actions and not _is_reasonable_pe(candidate):
                self.bytez = original_bytez
            else:
                self.bytez = candidate

        self.current_step += 1

        pred = self._predict(self.bytez)
        is_evaded = not pred.get("malicious", True)
        score = float(pred.get("score", 0.0))
        self._current_episode_record["final_score"] = score
        self._current_episode_record["steps"] = self.current_step

        reward = self.reward_fn(
            original_score=self.original_score,
            score=score,
            is_evaded=is_evaded,
            step=self.current_step,
            max_steps=self.max_turns,
        )

        terminated = is_evaded
        truncated = self.current_step >= self.max_turns

        if is_evaded:
            evaded_sha = sha256_bytes(self.bytez)
            self._current_episode_record["evaded"] = True
            self._current_episode_record["evaded_sha256"] = evaded_sha
            with open(os.path.join(self.output_dir, evaded_sha), "wb") as f:
                f.write(self.bytez)

        observation = self.feature_extractor.extract(self.bytez)
        info = {
            "action_name": action_obj.name,
            "is_evaded": is_evaded,
            "score": score,
            "sha256": sha256_bytes(self.bytez),
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        actions = self._current_episode_record.get("actions", []) if self._current_episode_record else []
        print(f"step={self.current_step} actions={actions}")


class MalwareEnv(BaseMalwareEnv):
    def __init__(self, **kwargs):
        super().__init__(reward_fn=SparseReward(), **kwargs)


class MalwareScoreEnv(BaseMalwareEnv):
    def __init__(self, **kwargs):
        super().__init__(reward_fn=ShapedReward(), **kwargs)
