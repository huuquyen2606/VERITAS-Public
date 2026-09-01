#!/usr/bin/env python
from __future__ import annotations

import argparse
import configparser
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dqeaf.classifier import load_classifier_artifact
from dqeaf.dataset import load_binary_corpus
from dqeaf.env import MalwareEnv
from dqeaf.network import DQEAFNet


def _load_ini(path: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    loaded = cfg.read(path, encoding="utf-8")
    if not loaded:
        raise FileNotFoundError(f"Config file not found: {path}")
    return cfg


def _get_str(cli_value: str | None, cfg: configparser.ConfigParser, section: str, key: str, default: str | None = None) -> str | None:
    if cli_value is not None:
        return cli_value
    value = cfg.get(section, key, fallback=default)
    if value is None:
        return None
    value = value.strip()
    return value if value != "" else None


def _get_int(cli_value: int | None, cfg: configparser.ConfigParser, section: str, key: str, default: int) -> int:
    if cli_value is not None:
        return cli_value
    return cfg.getint(section, key, fallback=default)


def _parse_csv(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained DQEAF model")
    p.add_argument("--config", default="configure/config.ini", help="Path to config ini")
    p.add_argument("--model-dir", default=None, help="Directory containing q_target.pt and classifier")
    p.add_argument("--malicious-dir", default=None, help="Directory with malicious test files")
    p.add_argument("--max-turn", type=int, default=None)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--device", choices=["cpu", "cuda"], default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ini = _load_ini(args.config)

    model_dir = _get_str(args.model_dir, ini, "Eval", "model_dir", "outputs/dqeaf")
    malicious_dir = _get_str(args.malicious_dir, ini, "Dataset", "malicious_test_dir")
    malicious_exclude_dirs = _parse_csv(_get_str(None, ini, "Dataset", "malicious_test_exclude_dirs", "Benign"))
    max_turn = _get_int(args.max_turn, ini, "Eval", "max_turn", 80)
    max_files = _get_int(args.max_files, ini, "Eval", "max_files", 200)
    device = _get_str(args.device, ini, "Runtime", "device", "cpu")

    if model_dir is None:
        raise RuntimeError("Missing model_dir. Set Eval.model_dir in config.ini or pass --model-dir")
    if malicious_dir is None:
        raise RuntimeError("Missing malicious test path. Set Dataset.malicious_test_dir in config.ini or pass --malicious-dir")

    model_dir_path = Path(model_dir)
    classifier = load_classifier_artifact(model_dir_path)

    q_target = DQEAFNet(input_dim=513, action_dim=4)
    q_target.load_state_dict(torch.load(model_dir_path / "q_target.pt", map_location=device))
    q_target.to(device)
    q_target.eval()

    samples = load_binary_corpus(
        malicious_dir,
        max_files,
        exclude_dir_names=malicious_exclude_dirs,
    )
    env = MalwareEnv(samples, classifier, max_turn=max_turn, seed=123)

    success = 0
    total_reward = 0.0

    with torch.no_grad():
        for sample in samples:
            state = env.reset_with_sample(sample)
            episode_success = False
            for _ in range(max_turn):
                s = torch.from_numpy(state).float().unsqueeze(0).to(device)
                action = int(torch.argmax(q_target(s), dim=1).item())
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if reward > 0:
                    episode_success = True
                if terminated or truncated:
                    break
            if episode_success:
                success += 1

    sr = (success / max(1, len(samples))) * 10.0
    result = {
        "num_samples": len(samples),
        "success": success,
        "SR": sr,
        "avg_reward": total_reward / max(1, len(samples)),
        "config": args.config,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
