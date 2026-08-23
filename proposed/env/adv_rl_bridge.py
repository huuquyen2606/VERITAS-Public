from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
import uuid
from typing import Any, Dict, Optional

import numpy as np
import redis


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADV_REPO_ROOT = PROJECT_ROOT / "env" / "adv_RL_env"
ADV_SYSTEM_ROOT = ADV_REPO_ROOT / "malware_rl_system"
for import_root in (PROJECT_ROOT, ADV_REPO_ROOT, ADV_SYSTEM_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from core.redis_orchestrator import RedisOrchestrator  
from evaluation.detector_api import DetectorEvaluator  
from evaluation.reward_calculator import RewardCalculator, is_multiclass_evaded  
from features.state_builder import AdvStateBuilder  


@dataclass
class AdvRLBaselineResult:
    original_name: str
    true_label: str
    episode_id: str
    redis_key: str
    redis_data: Dict[str, str]
    state: np.ndarray
    sandbox_context: Dict[str, Any]
    static_source: str


@dataclass
class AdvRLEvaluationResult:
    original_name: str
    candidate_name: str
    true_label: str
    reward: Optional[float]
    integrity_score: float
    functionality_score: float
    detector_scores: Dict[str, float]
    evaded: bool
    redis_key: str
    redis_data: Dict[str, str]
    state: np.ndarray
    sandbox_context: Dict[str, Any]
    static_source: str


class AdvRLPipelineBridge:
    """Thin integration layer around the Adv-RL Env Redis pipeline."""

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        work_dir: str | os.PathLike[str] = "/tmp",
        poll_interval: float = 2.0,
        max_wait_sec: int = 1800,
        run_detector: bool = True,
        run_functionality_inline: bool = False,
    ) -> None:
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True,
        )
        self.orchestrator = RedisOrchestrator(host=redis_host, port=redis_port, db=redis_db)
        self.work_dir = Path(work_dir)
        self.poll_interval = poll_interval
        self.max_wait_sec = max_wait_sec
        self.run_detector = run_detector
        self.run_functionality_inline = run_functionality_inline
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db

        self.reward_calculator = RewardCalculator(host=redis_host, port=redis_port, db=redis_db)
        self.state_builder = AdvStateBuilder(max_steps=5)
        self.detector_evaluator: Optional[DetectorEvaluator] = None
        self.detector_loaded = False
        self.functionality_evaluator: Optional[Any] = None

    def ping(self) -> bool:
        try:
            return bool(self.redis_client.ping())
        except redis.RedisError:
            return False

    def evaluate_pair(
        self,
        original_path: str | os.PathLike[str],
        candidate_path: str | os.PathLike[str],
        true_label: str,
        episode_id: Optional[str] = None,
        refresh_baseline: bool = False,
    ) -> AdvRLEvaluationResult:
        """Run original/candidate through the Adv-RL Env pipeline."""
        baseline = self.prepare_original(
            original_path=original_path,
            true_label=true_label,
            episode_id=episode_id,
            refresh_baseline=refresh_baseline,
        )
        return self.evaluate_candidate(
            baseline=baseline,
            candidate_path=candidate_path,
        )

    def prepare_original(
        self,
        original_path: str | os.PathLike[str],
        true_label: str,
        episode_id: Optional[str] = None,
        refresh_baseline: bool = False,
    ) -> AdvRLBaselineResult:
        """Stage and extract the original sample once for an episode."""
        if not self.ping():
            raise RuntimeError("Redis is not reachable. Start adv_RL_env Redis/workers first.")

        episode_id = episode_id or f"veritas_{uuid.uuid4().hex[:10]}"
        original_name = self._stage_file(original_path, prefix="ori")
        baseline_key = f"baseline:{original_name}:data"
        baseline_exists = bool(self.redis_client.exists(baseline_key))
        self.redis_client.hset(baseline_key, "true_label", true_label)

        if refresh_baseline or not baseline_exists:
            self._dispatch_and_wait(
                target_file_name=original_name,
                parent_file_name=original_name,
                is_original=True,
                episode_id=episode_id,
            )
        else:
            logger.debug("Using cached baseline data for %s.", original_name)

        redis_data = self.redis_client.hgetall(baseline_key)
        state_result = self.state_builder.build_from_redis_hash(
            redis_data,
            true_label=true_label,
            pe_path=self.work_dir / original_name,
        )

        return AdvRLBaselineResult(
            original_name=original_name,
            true_label=true_label,
            episode_id=episode_id,
            redis_key=baseline_key,
            redis_data=redis_data,
            state=state_result.state,
            sandbox_context=state_result.sandbox_context,
            static_source=state_result.static_source,
        )

    def evaluate_candidate(
        self,
        baseline: AdvRLBaselineResult,
        candidate_path: str | os.PathLike[str],
    ) -> AdvRLEvaluationResult:
        """Evaluate one mutated candidate against a prepared original sample."""
        if not self.ping():
            raise RuntimeError("Redis is not reachable. Start adv_RL_env Redis/workers first.")

        true_label = baseline.true_label
        original_name = baseline.original_name
        episode_id = baseline.episode_id
        candidate_name = self._stage_file(candidate_path, prefix="adv")
        episode_key = f"episode:{candidate_name}:data"

        self._dispatch_and_wait(
            target_file_name=candidate_name,
            parent_file_name=original_name,
            is_original=False,
            episode_id=episode_id,
        )

        if self.run_detector:
            self._run_detector(candidate_name)

        if self.run_functionality_inline:
            self._run_functionality(candidate_name, original_name)
        else:
            self._wait_for_hash_fields(
                episode_key,
                required_fields=("integrity_score", "functionality_score"),
            )

        reward = self.reward_calculator.calculate_and_save(candidate_name, true_label)

        redis_data = self.redis_client.hgetall(episode_key)
        detector_scores = self._json_dict(redis_data.get("detector_scores", "{}"))
        integrity_score = float(redis_data.get("integrity_score", 0.0))
        functionality_score = float(redis_data.get("functionality_score", 0.0))
        evaded = is_multiclass_evaded(detector_scores, true_label)
        state_result = self.state_builder.build_from_redis_hash(
            redis_data,
            true_label=true_label,
            pe_path=self.work_dir / candidate_name,
        )

        return AdvRLEvaluationResult(
            original_name=original_name,
            candidate_name=candidate_name,
            true_label=true_label,
            reward=reward,
            integrity_score=integrity_score,
            functionality_score=functionality_score,
            detector_scores=detector_scores,
            evaded=evaded,
            redis_key=episode_key,
            redis_data=redis_data,
            state=state_result.state,
            sandbox_context=state_result.sandbox_context,
            static_source=state_result.static_source,
        )

    def _stage_file(self, path: str | os.PathLike[str], prefix: str) -> str:
        src = Path(path)
        if not src.is_file():
            raise FileNotFoundError(src)

        suffix = src.suffix or ".exe"
        staged_name = f"{prefix}_{uuid.uuid4().hex[:12]}{suffix}"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, self.work_dir / staged_name)
        return staged_name

    def _dispatch_and_wait(
        self,
        target_file_name: str,
        parent_file_name: str,
        is_original: bool,
        episode_id: str,
    ) -> None:
        self.orchestrator.dispatch_task(
            target_file_name=target_file_name,
            parent_file_name=parent_file_name,
            is_original=is_original,
            episode_id=episode_id,
        )

        prefix = "baseline" if is_original else "episode"
        barrier_key = f"{prefix}:{target_file_name}:tasks_remaining"
        self._wait_for_barrier(barrier_key)

    def _wait_for_barrier(self, barrier_key: str) -> None:
        deadline = time.time() + self.max_wait_sec
        while time.time() < deadline:
            value = self.redis_client.get(barrier_key)
            if value is not None and int(value) <= 0:
                return
            time.sleep(self.poll_interval)

        raise TimeoutError(f"Timed out waiting for Redis barrier {barrier_key}")

    def _run_detector(self, candidate_name: str) -> None:
        if self.detector_evaluator is None:
            self.detector_evaluator = DetectorEvaluator(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
            )
            self.detector_loaded = self.detector_evaluator.load_model()

        if not self.detector_loaded:
            logger.warning("Detector weights are missing; reward will use conservative P_original=1.0.")
            return

        self.detector_evaluator.evaluate_episode(candidate_name)

    def _run_functionality(self, candidate_name: str, original_name: str) -> None:
        if self.functionality_evaluator is None:
            from evaluation.functionality_evaluator import FunctionalityEvaluator

            self.functionality_evaluator = FunctionalityEvaluator(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
            )

        self.functionality_evaluator.evaluate(candidate_name, original_name)

    def _wait_for_hash_fields(
        self,
        redis_key: str,
        required_fields: tuple[str, ...],
    ) -> None:
        deadline = time.time() + self.max_wait_sec
        while time.time() < deadline:
            if all(self.redis_client.hexists(redis_key, field) for field in required_fields):
                return
            time.sleep(self.poll_interval)

        raise TimeoutError(
            f"Timed out waiting for Redis fields {required_fields} in {redis_key}"
        )

    @staticmethod
    def _json_dict(raw: str) -> Dict[str, float]:
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        if not isinstance(parsed, dict):
            return {}
        return {str(k): float(v) for k, v in parsed.items()}
