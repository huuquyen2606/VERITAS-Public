import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from obfumal.obfumal import (
    DQNAgent,

    LightGBMDetector,
    MalwareEnv,
    MalwareScoreEnv,
    default_action_set,
    summarize_history,
    set_seed,
    apply_ini_overrides,
)
from obfumal.utils.preflight import detector_details, probe_detector_on_samples
from obfumal.utils.sample_preparation import DEFAULT_SAMPLES_DIR, prepare_samples_from_dataset


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return apply_ini_overrides(config)


def build_detector(cfg: Dict[str, Any]):
    dtype = cfg.get("type", "lightgbm")
    if dtype == "lightgbm":
        return LightGBMDetector(
            model_path=cfg["model_path"],
            threshold=cfg.get("threshold", 0.9),
            feature_backend=cfg.get("feature_backend", "auto"),
        )

    raise ValueError(f"Unsupported detector type: {dtype}")


def build_env(cfg: Dict[str, Any], detector, mode: str):
    env_cls = MalwareScoreEnv if mode == "score" else MalwareEnv
    action_set = default_action_set(
        include_obfuscation=cfg.get("include_obfuscation", True),
        upx_path=cfg.get("upx_path", "upx"),
        darkarmour_path=cfg.get("darkarmour_path", "darkarmour"),
    )
    return env_cls(
        detector=detector,
        sample_dir=cfg["sample_dir"],
        sha256_list=cfg.get("sha256_list"),
        max_turns=cfg.get("max_turns", 5),
        random_sample=cfg.get("random_sample", False),
        cache_samples=cfg.get("cache_samples", False),
        cache_predictions=cfg.get("cache_predictions", True),
        skip_benign=cfg.get("skip_benign", True),
        max_reset_attempts=cfg.get("max_reset_attempts"),
        validate_actions=cfg.get("validate_actions", True),
        isolate_native_actions=cfg.get("isolate_native_actions", True),
        action_apply_timeout_sec=cfg.get("action_apply_timeout_sec", 2.0),
        output_dir=cfg.get("output_dir", "artifacts/evaded/blackbox"),
        action_set=action_set,
    )


def run_preflight(detector, env, seed: int):
    det = detector_details(detector)
    print(
        f"[PRECHECK] detector={det['name']} | model={det['model_path']} | "
        f"threshold={det['threshold']} | feature_source={det['feature_source']}"
    )
    stats = probe_detector_on_samples(
        detector=detector,
        sample_store=env.sample_store,
        sha256_list=env.available_sha256,
        sample_size=min(8, len(env.available_sha256)),
        seed=seed,
    )
    print(
        f"[PRECHECK] sampled={stats['sampled']} | malicious={stats['malicious_count']} | "
        f"errors={stats['errors']} | score_min={stats['score_min']} | "
        f"score_max={stats['score_max']} | score_mean={stats['score_mean']}"
    )
    if env.skip_benign and stats["sampled"] > 0 and stats["malicious_count"] == 0:
        raise RuntimeError(
            "Precheck failed: all sampled files are predicted benign while skip_benign=True. "
            "Lower threshold, disable skip_benign, or fix feature backend/model mismatch."
        )


def prepare_test_samples(args: argparse.Namespace, env_cfg: Dict[str, Any]) -> None:
    if args.sample_dir:
        env_cfg["sample_dir"] = args.sample_dir

    if not args.test_dir:
        return

    target_sample_dir = env_cfg.get("sample_dir") or str(DEFAULT_SAMPLES_DIR)
    summary = prepare_samples_from_dataset(
        source_dir=args.test_dir,
        dest_dir=target_sample_dir,
        excluded_dir_names=args.exclude_dirs,
        clear_destination=not args.keep_existing_samples and not args.recreate_sample_dir,
        recreate_destination=args.recreate_sample_dir,
    )
    env_cfg["sample_dir"] = str(summary.dest_dir)

    print("[PREP:EVAL] prepared evaluation samples")
    print(f"[PREP:EVAL] source={summary.source_dir}")
    print(f"[PREP:EVAL] destination={summary.dest_dir}")
    print(f"[PREP:EVAL] destination_recreated={summary.recreated_destination}")
    print(f"[PREP:EVAL] cleared_existing={summary.cleared_existing}")
    print(f"[PREP:EVAL] scanned={summary.scanned_files} executable_candidates={summary.executable_candidates}")
    print(
        f"[PREP:EVAL] copied={summary.copied_files} "
        f"skipped_existing={summary.skipped_existing} "
        f"skipped_non_executable={summary.skipped_non_executable} failed={summary.failed}"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate DQN Agent for OBFU-mal")
    parser.add_argument("--config", type=str, default="configs/evaluation.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model")
    parser.add_argument("--mode", type=str, choices=["blackbox"], default=None, help="Reward mode override")
    parser.add_argument(
        "--test_dir",
        type=str,
        default=None,
        help="Raw evaluation dataset directory (auto scan subfolders, skip Benign, rename by SHA256).",
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        default=None,
        help="Directory that stores hash-named samples used by the environment.",
    )
    parser.add_argument(
        "--exclude_dirs",
        nargs="*",
        default=["Benign"],
        help="Folder names to ignore while scanning --test_dir.",
    )
    sample_reset_group = parser.add_mutually_exclusive_group()
    sample_reset_group.add_argument(
        "--keep_existing_samples",
        action="store_true",
        help="Append prepared files to existing sample_dir instead of clearing old hash-named files.",
    )
    sample_reset_group.add_argument(
        "--recreate_sample_dir",
        action="store_true",
        help="Delete sample_dir and create it again before preparing samples.",
    )
    parser.add_argument(
        "--plain-log",
        action="store_true",
        help="Print one line per sample (no tqdm carriage-return rendering).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("seed", 123))
    set_seed(seed)

    env_cfg = config["env"]
    det_cfg = config["detector"]
    agent_cfg = config["agent"]
    eval_cfg = config.get("evaluation", {})

    if args.mode is not None:
        env_cfg["mode"] = args.mode
    env_mode = env_cfg.get("mode", "blackbox")
    if env_mode != "blackbox":
        raise ValueError("Evaluation is restricted to blackbox mode so the environment uses SparseReward only.")

    prepare_test_samples(args, env_cfg)

    detector = build_detector(det_cfg)
    env = build_env(env_cfg, detector, env_mode)
    run_preflight(detector, env, seed=seed)

    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=agent_cfg.get("lr", 1e-3),
        gamma=agent_cfg.get("gamma", 0.99),
        batch_size=agent_cfg.get("batch_size", 32),
        buffer_size=agent_cfg.get("buffer_size", 10000),
        epsilon_start=agent_cfg.get("epsilon_start", 1.0),
        epsilon_end=agent_cfg.get("epsilon_end", 0.05),
        epsilon_decay_steps=agent_cfg.get("epsilon_decay_steps", 10000),
        target_update_freq=agent_cfg.get("target_update_freq", 100),
        hidden_dims=agent_cfg.get("hidden_dims"),
        device=agent_cfg.get("device", "cpu"),
        policy=agent_cfg.get("policy", "boltzmann"),
        temperature=agent_cfg.get("temperature", 1.0),
    )

    model_path = args.model or config["training"]["save_path"]
    agent.load(model_path)

    episodes = eval_cfg.get("episodes")
    if not episodes or episodes <= 0:
        episodes = len(env.available_sha256)
    total_samples = len(env.available_sha256)
    print(f"[EVAL] episodes={episodes} | corpus_size={total_samples} | max_turns={env.max_turns}")
    pbar = tqdm(
        range(episodes),
        dynamic_ncols=True,
        disable=args.plain_log or not sys.stdout.isatty(),
    )
    for episode in pbar:
        state, info = env.reset()
        done = False
        truncated = False
        last_step_info: Dict[str, Any] = {}

        while not (done or truncated):
            action = agent.select_action(state, evaluate=True)
            state, _, done, truncated, step_info = env.step(action)
            last_step_info = step_info

        sample_idx = (episode % total_samples) + 1
        sample_sha = str(info.get("sha256", ""))[:10]
        evaded = "Y" if last_step_info.get("is_evaded", False) else "N"
        score = float(last_step_info.get("score", info.get("original_score", 0.0)))
        msg = (
            f"[EVAL] sample {sample_idx}/{total_samples} | ep {episode + 1}/{episodes} | "
            f"sha {sample_sha} | steps {env.current_step}/{env.max_turns} | evaded {evaded} | score {score:.4f}"
        )
        if args.plain_log:
            print(msg, flush=True)
        else:
            pbar.set_description(msg)

    run_dir = eval_cfg.get("run_dir", "artifacts/runs")
    os.makedirs(run_dir, exist_ok=True)
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = os.path.join(run_dir, f"eval_history_{run_id}.json")
    history_payload = {
        "by_sample": env.history,
        "by_episode": env.episode_history,
        "stats": env.run_stats,
    }
    with open(history_path, "w") as f:
        json.dump(history_payload, f, indent=2)

    summary = summarize_history(
        env.history,
        episode_history=env.episode_history,
        prepared_total=len(env.available_sha256),
        skipped_benign=env.run_stats.get("skipped_benign", 0),
    )
    report_path = os.path.join(run_dir, f"eval_summary_{run_id}.json")
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"History saved to {history_path}")
    print(f"Summary saved to {report_path}")


if __name__ == "__main__":
    main()
