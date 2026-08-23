"""Evaluate a trained VERITAS agent and optionally save adversarial examples."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.veritas_agent import VERITASAgent  
from env.malware_env import MalwareEnv  
from scripts.rl_entrypoint_utils import (  
    action_name,
    choose_device,
    discover_samples,
    env_kwargs_from_config,
    episode_count,
    limit_samples_per_family,
    load_checkpoint,
    load_yaml_config,
    make_agent_kwargs,
    resolve_path,
    set_global_seed,
    summarize_action_usage,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VERITAS-Proposal agent.")
    parser.add_argument("--config", default="configs/test.yaml", help="YAML config path.")
    parser.add_argument("--checkpoint", default=None, help="Override test.checkpoint.")
    parser.add_argument("--episodes", default=None, help="Override test.episodes; int or 'all'.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for smoke runs.")
    parser.add_argument("--samples-per-family", type=int, default=None, help="Use at most N samples from each family.")
    parser.add_argument("--output-dir", default=None, help="Override test.output_dir.")
    parser.add_argument("--run-id", default=None, help="Shard/run identifier for parallel evaluation output isolation.")
    parser.add_argument("--resume-existing", action="store_true", help="Skip samples whose trace JSON already exists and include them in final metrics.")
    parser.add_argument("--skip-indices", default="", help="Comma-separated 1-based sample indices to record as failed traces and skip.")
    parser.add_argument("--device", default=None, help="Override agent.device.")
    parser.add_argument("--work-dir", default="/tmp", help="Shared staging directory used by Adv-RL workers.")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--poll-interval", type=float, default=None)
    parser.add_argument("--max-wait-sec", type=int, default=None)
    parser.add_argument("--skip-detector", action="store_true", help="Skip MAttnHealth inference.")
    parser.add_argument("--inline-functionality", action="store_true", help="Run functionality evaluator in-process.")
    parser.add_argument("--prepare-cfg-calls", action=argparse.BooleanOptionalAction, default=True, help="angr call-site prep for actions 17/18 (default: ON).")
    parser.add_argument("--evaluate-reset", action="store_true", help="Extract original sample during reset.")
    return parser.parse_args()


def _normalize_run_id(raw: str | None) -> str | None:
    if raw is None:
        return None
    run_id = raw.strip()
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
    if not run_id or any(ch not in allowed for ch in run_id):
        raise ValueError("--run-id may only contain letters, numbers, underscore, hyphen, and dot.")
    if run_id in {".", ".."}:
        raise ValueError("--run-id cannot be '.' or '..'.")
    return run_id


def _trace_path(trace_dir: Path, idx: int, label: str, sample_path: Path) -> Path:
    return trace_dir / f"{idx:06d}__{label}__{sample_path.stem}.json"


def _parse_skip_indices(raw: str | None) -> set[int]:
    if not raw:
        return set()
    indices: set[int] = set()
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError("--skip-indices values must be positive 1-based indices.")
        indices.add(value)
    return indices


def _read_existing_trace(path: Path, *, idx: int, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("index", -1)) != int(idx):
        raise ValueError(f"Existing trace index mismatch for {path}: {payload.get('index')} != {idx}")
    if str(payload.get("label")) != str(label):
        raise ValueError(f"Existing trace label mismatch for {path}: {payload.get('label')} != {label}")
    return payload


def _accumulate_trace_metrics(
    trace_payload: dict[str, Any],
    *,
    family_total: Counter[str],
    family_evaded: Counter[str],
    action_counter: Counter[int],
    reward_values: list[float],
    steps_to_evade: list[int],
) -> None:
    label = str(trace_payload.get("label"))
    family_total[label] += 1

    if bool(trace_payload.get("evaded", False)):
        family_evaded[label] += 1
        steps_to_evade.append(int(trace_payload.get("steps") or 0))

    for step in trace_payload.get("trace") or []:
        if "action" in step and step.get("action") is not None:
            action_counter[int(step["action"])] += 1
        if "reward" in step and step.get("reward") is not None:
            reward_values.append(float(step["reward"]))


def main() -> int:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    test_cfg = cfg.get("test", {})
    train_cfg = cfg.get("train", {})
    agent_cfg = cfg.get("agent", {})
    run_id = _normalize_run_id(args.run_id or test_cfg.get("run_id"))

    seed = int(train_cfg.get("seed", 1337))
    set_global_seed(seed)
    device = choose_device(args.device or agent_cfg.get("device", "cpu"))

    samples = discover_samples(cfg["dataset"])
    samples = limit_samples_per_family(samples, args.samples_per_family)
    total = episode_count(args.episodes or test_cfg.get("episodes", "all"), len(samples))
    total = min(total, len(samples))
    if args.limit is not None:
        total = min(total, int(args.limit))
    if total <= 0:
        print(
            "[+] No test episodes requested; "
            f"config OK, samples={len(samples)}, device={device}."
        )
        return 0
    samples = samples[:total]

    checkpoint_path = resolve_path(args.checkpoint or test_cfg.get("checkpoint"), must_exist=True)
    output_dir = resolve_path(args.output_dir or test_cfg.get("output_dir", "outputs/aes_test_run_001"))
    trace_dir = output_dir / "traces" / run_id if run_id else output_dir / "traces"
    ae_dir = output_dir / "aes"
    metrics_path = output_dir / "metrics" / f"{run_id}.json" if run_id else output_dir / "metrics.json"
    traces_path = None if run_id else output_dir / "traces.json"
    resume_existing = bool(args.resume_existing)
    if resume_existing and not test_cfg.get("save_traces", True):
        raise ValueError("--resume-existing requires test.save_traces=true")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if run_id:
        (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    if test_cfg.get("save_traces", True):
        trace_dir.mkdir(parents=True, exist_ok=True)
    if test_cfg.get("save_aes", True):
        ae_dir.mkdir(parents=True, exist_ok=True)
        if run_id:
            for family in cfg.get("dataset", {}).get("families", []):
                (ae_dir / str(family)).mkdir(parents=True, exist_ok=True)

    agent = VERITASAgent(**make_agent_kwargs(cfg, device))
    checkpoint = load_checkpoint(checkpoint_path, agent, device, load_optimizer=False)
    agent.policy_net.eval()
    agent.target_net.eval()

    env = MalwareEnv(**env_kwargs_from_config(cfg, args))
    greedy = bool(test_cfg.get("greedy", True))

    family_total: Counter[str] = Counter()
    family_evaded: Counter[str] = Counter()
    action_counter: Counter[int] = Counter()
    reward_values: list[float] = []
    steps_to_evade: list[int] = []
    traces: list[dict[str, Any]] = []
    started_at = time.time()

    print(
        "[+] Testing VERITAS: "
        f"samples={len(samples)} checkpoint={checkpoint_path} device={device} "
        f"greedy={greedy} run_id={run_id or 'none'} resume_existing={resume_existing}"
    )

    skip_indices = _parse_skip_indices(args.skip_indices)

    for idx, sample in enumerate(samples, start=1):
        sample_path = Path(sample["path"])
        label = str(sample["label"])
        trace_file = _trace_path(trace_dir, idx, label, sample_path)
        if resume_existing and trace_file.is_file():
            trace_payload = _read_existing_trace(trace_file, idx=idx, label=label)
            traces.append(trace_payload)
            _accumulate_trace_metrics(
                trace_payload,
                family_total=family_total,
                family_evaded=family_evaded,
                action_counter=action_counter,
                reward_values=reward_values,
                steps_to_evade=steps_to_evade,
            )
            print(
                f"[RESUME] {idx}/{len(samples)} label={label} "
                f"evaded={bool(trace_payload.get('evaded', False))} existing_trace={trace_file.name}"
            )
            continue

        if idx in skip_indices:
            trace_payload = {
                "index": idx,
                "run_id": run_id,
                "sample": str(sample_path.relative_to(PROJECT_ROOT) if sample_path.is_relative_to(PROJECT_ROOT) else sample_path),
                "label": label,
                "evaded": False,
                "total_reward": 0.0,
                "steps": 0,
                "error": f"manual_skip_index:{idx}",
                "skipped": True,
                "skip_reason": "Skipped during resume because this sample repeatedly caused a native segfault.",
                "trace": [],
            }
            traces.append(trace_payload)
            _accumulate_trace_metrics(
                trace_payload,
                family_total=family_total,
                family_evaded=family_evaded,
                action_counter=action_counter,
                reward_values=reward_values,
                steps_to_evade=steps_to_evade,
            )
            if test_cfg.get("save_traces", True):
                write_json(trace_file, trace_payload)
            print(f"[SKIP] {idx}/{len(samples)} label={label} trace={trace_file.name}")
            continue

        trace_steps: list[dict[str, Any]] = []
        total_reward = 0.0
        evaded = False
        error: str | None = None

        try:
            state, _ = env.reset(
                options={
                    "malware_path": str(sample_path),
                    "true_label": label,
                }
            )

            for _ in range(int(cfg.get("env", {}).get("max_steps", 5))):
                action = agent.select_action(state, eval_mode=greedy)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)

                total_reward += float(reward)

                step_record = {
                    "step": int(info.get("step", len(trace_steps) + 1)),
                    "action": int(action),
                    "agent_action": info.get("agent_action", int(action)),
                    "action_name": info.get("action_name") or action_name(int(action)),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "evaded": bool(info.get("evaded", False)),
                    "committed": bool(info.get("committed", False)),
                    "rolled_back": bool(info.get("rolled_back", False)),
                    "reward_source": info.get("reward_source"),
                    "integrity_score": info.get("integrity_score"),
                    "functionality_score": info.get("functionality_score"),
                    "detector_scores": info.get("detector_scores"),
                }
                trace_steps.append(step_record)

                state = next_state
                if done:
                    evaded = bool(info.get("evaded", False))
                    break

            if evaded:
                if test_cfg.get("save_aes", True):
                    if run_id:
                        family_dir = ae_dir / label
                        family_dir.mkdir(parents=True, exist_ok=True)
                        out_name = f"{run_id}__{sample_path.stem}__evaded.exe"
                        (family_dir / out_name).write_bytes(env.pe_bytes)
                    else:
                        out_name = f"{label}__{sample_path.stem}__evaded.exe"
                        (ae_dir / out_name).write_bytes(env.pe_bytes)

        except Exception as exc:
            error = str(exc)

        trace_payload = {
            "index": idx,
            "run_id": run_id,
            "sample": str(sample_path.relative_to(PROJECT_ROOT) if sample_path.is_relative_to(PROJECT_ROOT) else sample_path),
            "label": label,
            "evaded": evaded,
            "total_reward": total_reward,
            "steps": len(trace_steps),
            "error": error,
            "trace": trace_steps,
        }
        traces.append(trace_payload)
        _accumulate_trace_metrics(
            trace_payload,
            family_total=family_total,
            family_evaded=family_evaded,
            action_counter=action_counter,
            reward_values=reward_values,
            steps_to_evade=steps_to_evade,
        )
        if test_cfg.get("save_traces", True):
            write_json(trace_file, trace_payload)

        print(
            f"[TEST] {idx}/{len(samples)} label={label} evaded={evaded} "
            f"steps={len(trace_steps)} reward={total_reward:.2f} error={error}"
        )

    evaded_total = sum(family_evaded.values())
    per_family = {
        family: {
            "samples": int(count),
            "evaded": int(family_evaded[family]),
            "evasion_rate": float(family_evaded[family] / count) if count else 0.0,
        }
        for family, count in sorted(family_total.items())
    }

    metrics = {
        "run_id": run_id,
        "checkpoint": str(checkpoint_path),
        "checkpoint_episode": checkpoint.get("episode") if isinstance(checkpoint, dict) else None,
        "samples": len(traces),
        "evaded": int(evaded_total),
        "evasion_rate": float(evaded_total / len(traces)) if traces else 0.0,
        "mean_steps_to_evade": float(sum(steps_to_evade) / len(steps_to_evade)) if steps_to_evade else None,
        "per_family_evasion": per_family,
        "action_usage_freq": summarize_action_usage(action_counter.elements()),
        "reward_distribution": {
            "count": len(reward_values),
            "min": min(reward_values) if reward_values else None,
            "max": max(reward_values) if reward_values else None,
            "mean": float(sum(reward_values) / len(reward_values)) if reward_values else None,
        },
        "elapsed_sec": round(time.time() - started_at, 3),
    }

    write_json(metrics_path, metrics)
    if traces_path is not None and test_cfg.get("save_traces", True):
        write_json(traces_path, {"traces": traces})

    print(f"[+] Test complete. metrics={metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
