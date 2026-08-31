import argparse
import datetime as dt
import json
import os
from typing import Dict, Any

import yaml
from tqdm import tqdm

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


def subset_actions(subset: str, env_cfg: Dict[str, Any]):
    actions = default_action_set(
        include_obfuscation=True,
        upx_path=env_cfg.get("upx_path", "upx"),
        darkarmour_path=env_cfg.get("darkarmour_path", "darkarmour"),
    )
    if subset == "all":
        return actions
    if subset == "classic":
        return [a for a in actions if not a.name.startswith("DarkarmourXOR")]
    if subset == "obfuscation":
        return [a for a in actions if a.name.startswith("DarkarmourXOR") or a.name == "UPXPack"]
    raise ValueError(f"Unknown subset: {subset}")


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


def main():
    parser = argparse.ArgumentParser(description="Ablation evaluation")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model")
    parser.add_argument("--subset", type=str, choices=["all", "classic", "obfuscation"], default="all")
    parser.add_argument("--mode", type=str, choices=["blackbox", "score"], default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("seed", 123))
    set_seed(seed)

    env_cfg = config["env"]
    det_cfg = config["detector"]
    agent_cfg = config["agent"]

    if args.mode is not None:
        env_cfg["mode"] = args.mode

    detector = build_detector(det_cfg)
    env_cls = MalwareScoreEnv if env_cfg.get("mode", "blackbox") == "score" else MalwareEnv
    env = env_cls(
        detector=detector,
        sample_dir=env_cfg["sample_dir"],
        sha256_list=env_cfg.get("sha256_list"),
        max_turns=env_cfg.get("max_turns", 5),
        random_sample=False,
        cache_samples=env_cfg.get("cache_samples", False),
        cache_predictions=env_cfg.get("cache_predictions", True),
        skip_benign=env_cfg.get("skip_benign", True),
        max_reset_attempts=env_cfg.get("max_reset_attempts"),
        validate_actions=env_cfg.get("validate_actions", True),
        isolate_native_actions=env_cfg.get("isolate_native_actions", True),
        action_apply_timeout_sec=env_cfg.get("action_apply_timeout_sec", 2.0),
        output_dir=env_cfg.get("output_dir", "artifacts/evaded/blackbox"),
        action_set=subset_actions(args.subset, env_cfg),
    )
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
    )

    model_path = args.model or config["training"]["save_path"]
    agent.load(model_path)

    total_samples = len(env.available_sha256)
    print(f"[ABLATE:{args.subset}] episodes={total_samples} | corpus_size={total_samples} | max_turns={env.max_turns}")
    pbar = tqdm(range(total_samples), ncols=140)
    for episode in pbar:
        state, info = env.reset()
        done = False
        truncated = False
        last_step_info: Dict[str, Any] = {}
        while not (done or truncated):
            action = agent.select_action(state, evaluate=True)
            state, _, done, truncated, step_info = env.step(action)
            last_step_info = step_info

        sample_sha = str(info.get("sha256", ""))[:10]
        evaded = "Y" if last_step_info.get("is_evaded", False) else "N"
        score = float(last_step_info.get("score", info.get("original_score", 0.0)))
        pbar.set_description(
            f"[ABLATE:{args.subset}] sample {episode + 1}/{total_samples} | sha {sample_sha} | steps {env.current_step}/{env.max_turns} | evaded {evaded} | score {score:.4f}"
        )

    run_dir = "artifacts/runs"
    os.makedirs(run_dir, exist_ok=True)
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = os.path.join(run_dir, f"ablate_{args.subset}_{run_id}.json")
    with open(history_path, "w") as f:
        json.dump(env.history, f, indent=2)

    summary = summarize_history(env.history)
    report_path = os.path.join(run_dir, f"ablate_{args.subset}_{run_id}_summary.json")
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"History saved to {history_path}")
    print(f"Summary saved to {report_path}")


if __name__ == "__main__":
    main()
