"""Shared helpers for VERITAS train/test entrypoints."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read config files. Install requirements.txt "
            "or run: pip install PyYAML"
        ) from exc

    cfg_path = resolve_path(path, must_exist=True)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {cfg_path}")
    return data


def resolve_path(path: str | Path | None, *, must_exist: bool = False) -> Path:
    if path is None:
        raise ValueError("path cannot be None")
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p = p.resolve()
    if must_exist and not p.exists():
        raise FileNotFoundError(p)
    return p


def optional_path(path: str | Path | None) -> Path | None:
    if path in (None, ""):
        return None
    return resolve_path(path)


def path_list(value: Any) -> list[Path]:
    if value in (None, ""):
        return []
    if isinstance(value, (str, Path)):
        return [resolve_path(value)]
    return [resolve_path(item) for item in value]


def choose_device(requested: str | None) -> torch.device:
    requested = requested or "cpu"
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_samples(dataset_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    root = resolve_path(dataset_cfg["malware_root"], must_exist=True)
    families = list(dataset_cfg.get("families") or [])
    if not families:
        families = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "Benign")

    samples: list[dict[str, Any]] = []
    for family in families:
        family_dir = root / family
        if not family_dir.is_dir():
            print(f"[WARN] Missing family directory, skipped: {family_dir}")
            continue
        for path in sorted(family_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".exe", ".dll"}:
                samples.append({"path": path, "label": family})

    if not samples:
        raise RuntimeError(f"No PE samples found under {root} for families={families}")
    return samples


def limit_samples_per_family(samples: list[dict[str, Any]], count: int | None) -> list[dict[str, Any]]:
    if count is None:
        return samples
    count = int(count)
    if count <= 0:
        return []

    kept: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for sample in samples:
        label = str(sample["label"])
        current = seen.get(label, 0)
        if current >= count:
            continue
        kept.append(sample)
        seen[label] = current + 1
    return kept


def write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def make_agent_kwargs(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    agent_cfg = cfg.get("agent", {})
    return {
        "num_actions": int(agent_cfg.get("num_actions", 19)),
        "num_atoms": int(agent_cfg.get("num_atoms", 51)),
        "v_min": float(agent_cfg.get("v_min", -500.0)),
        "v_max": float(agent_cfg.get("v_max", 200.0)),
        "sigma_init": float(agent_cfg.get("sigma_init", 0.5)),
        "gamma": float(agent_cfg.get("gamma", 0.99)),
        "lr": float(agent_cfg.get("lr", 1e-4)),
        "adam_eps": float(agent_cfg.get("adam_eps", 1.5e-4)),
        "grad_clip": float(agent_cfg.get("grad_clip", 10.0)),
        "device": device,
    }


def make_buffer_kwargs(cfg: dict[str, Any], obs_dim: int) -> dict[str, Any]:
    per_cfg = cfg.get("agent", {}).get("per", {})
    return {
        "capacity": int(per_cfg.get("capacity", 10_000)),
        "obs_dim": int(obs_dim),
        "alpha": float(per_cfg.get("alpha", 0.6)),
        "beta": float(per_cfg.get("beta", 0.4)),
        "beta_increment": float(per_cfg.get("beta_increment", 0.001)),
        "epsilon": float(per_cfg.get("epsilon", 1e-5)),
    }


def env_kwargs_from_config(cfg: dict[str, Any], args: Any) -> dict[str, Any]:
    env_cfg = cfg.get("env", {})
    sandbox_cfg = cfg.get("sandbox", {})
    precomputed_cfg = cfg.get("precomputed", {})

    cr_dirs = path_list(precomputed_cfg.get("cr_exe_dir"))
    ct_dirs = path_list(precomputed_cfg.get("ct_exe_dir"))

    return {
        "max_steps": int(env_cfg.get("max_steps", 5)),
        "function_threshold": float(env_cfg.get("function_threshold", 80.0)),
        "evaluate_reset": bool(getattr(args, "evaluate_reset", False)),
        "prepare_cfg_calls": bool(getattr(args, "prepare_cfg_calls", False)),
        "cr_exe_dirs": cr_dirs,
        "ct_exe_dirs": ct_dirs,
        "redis_host": getattr(args, "redis_host", "localhost"),
        "redis_port": int(getattr(args, "redis_port", 6379)),
        "redis_db": int(getattr(args, "redis_db", 0)),
        "work_dir": getattr(args, "work_dir", "/tmp"),
        "poll_interval": float(
            getattr(args, "poll_interval", None) or sandbox_cfg.get("poll_interval", 2.0)
        ),
        "max_wait_sec": int(
            getattr(args, "max_wait_sec", None) or sandbox_cfg.get("timeout_full", 1800)
        ),
        "run_detector": not bool(getattr(args, "skip_detector", False)),
        "run_functionality_inline": bool(getattr(args, "inline_functionality", False)),
    }


def save_checkpoint(
    path: Path,
    *,
    agent: Any,
    cfg: dict[str, Any],
    episode: int,
    global_step: int,
    stats: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "episode": int(episode),
        "global_step": int(global_step),
        "config": cfg,
        "stats": stats,
        "policy_state_dict": agent.policy_net.state_dict(),
        "target_state_dict": agent.target_net.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
    }
    torch.save(payload, path)


def _check_checkpoint_action_head(path: Path, state_dict: dict[str, Any], agent: Any) -> None:
    weight = state_dict.get("advantage_stream.mu_weight")
    if not isinstance(weight, torch.Tensor):
        return

    num_actions = int(getattr(agent, "num_actions", 0) or 0)
    num_atoms = int(getattr(agent, "num_atoms", 0) or 0)
    if num_actions <= 0 or num_atoms <= 0:
        return

    actual_out = int(weight.shape[0])
    expected_out = num_actions * num_atoms
    if actual_out == expected_out:
        return

    if actual_out % num_atoms == 0:
        actual_actions = str(actual_out // num_atoms)
    else:
        actual_actions = "unknown"

    raise ValueError(
        f"Checkpoint action head is incompatible: {path} has "
        f"advantage_stream.mu_weight out_features={actual_out} "
        f"(num_actions={actual_actions} with num_atoms={num_atoms}), "
        f"but the current agent expects out_features={expected_out} "
        f"({num_actions} actions x {num_atoms} atoms). Train a fresh "
        "checkpoint or resume from a checkpoint created with the same action space."
    )


def load_checkpoint(path: Path, agent: Any, device: torch.device, *, load_optimizer: bool = False) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "policy_state_dict" in checkpoint:
        _check_checkpoint_action_head(path, checkpoint["policy_state_dict"], agent)
        agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        if "target_state_dict" in checkpoint:
            _check_checkpoint_action_head(path, checkpoint["target_state_dict"], agent)
            agent.target_net.load_state_dict(checkpoint["target_state_dict"])
        else:
            agent.sync_target_network()
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        _check_checkpoint_action_head(path, checkpoint["state_dict"], agent)
        agent.policy_net.load_state_dict(checkpoint["state_dict"])
        agent.sync_target_network()
        return checkpoint

    if isinstance(checkpoint, dict):
        _check_checkpoint_action_head(path, checkpoint, agent)
        agent.policy_net.load_state_dict(checkpoint)
        agent.sync_target_network()
        return {"format_version": 0}

    raise ValueError(f"Unsupported checkpoint format: {path}")


def episode_count(value: Any, total_samples: int) -> int:
    if value is None:
        return total_samples
    if isinstance(value, str) and value.lower() == "all":
        return total_samples
    return int(value)


def action_name(action_idx: int) -> str:
    try:
        from env.malware_env import agent_action_name

        return agent_action_name(int(action_idx))
    except Exception:
        return str(action_idx)


def summarize_action_usage(actions: Iterable[int]) -> dict[str, int]:
    counter = Counter(int(a) for a in actions)
    return {action_name(idx): count for idx, count in sorted(counter.items())}
