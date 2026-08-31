#!/usr/bin/env python
from __future__ import annotations

import argparse
import configparser
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / "dqeaf-venv" / "bin" / "python3"
if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != (ROOT / "dqeaf-venv").resolve():
    print(f"[*] Redirecting execution to {VENV_PYTHON} to use project dependencies...")
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dqeaf.classifier import (
    IndependentClassifier,
    LightGBMEmberClassifier,
    ScikitLearnEmberClassifier,
)
from dqeaf.config import DQEAFConfig
from dqeaf.dataset import load_binary_corpus, train_test_split_bytes
from dqeaf.train import Trainer


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


def _get_float(cli_value: float | None, cfg: configparser.ConfigParser, section: str, key: str, default: float) -> float:
    if cli_value is not None:
        return cli_value
    return cfg.getfloat(section, key, fallback=default)


def _parse_csv(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _sanitize_output_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return sanitized or "dqeaf"


def _derive_output_dir(train_dir: str, configured_output_dir: str | None) -> str:
    if configured_output_dir is not None:
        return configured_output_dir

    family_name = _sanitize_output_name(Path(train_dir).resolve().name)
    return str(Path("outputs") / f"{family_name}_dqeaf")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train DQEAF reproduction from paper")
    p.add_argument("--config", default="configure/config.ini", help="Path to config ini")
    p.add_argument(
        "--train_dir",
        "--train-dir",
        dest="train_dir",
        default=None,
        help="Directory containing malicious PE files for DQEAF training",
    )
    p.add_argument("--malicious-dir", default=None, help="Legacy alias of --train_dir")
    p.add_argument("--benign-dir", default=None, help="Directory containing benign PE files")
    p.add_argument("--output-dir", default=None, help="Where to save model and logs")

    p.add_argument("--classifier-backend", choices=["gbdt", "lgbm_ember"], default=None)
    p.add_argument("--classifier-path", default=None, help="Optional pre-trained legacy raw-513 GBDT classifier (.joblib)")
    p.add_argument("--gbdt-model-path", default=None, help="Path to pre-trained EMBER GBDT model (.pkl)")
    p.add_argument("--gbdt-threshold", type=float, default=None, help="Threshold for EMBER GBDT model")
    p.add_argument("--lgbm-model-path", default=None, help="Path to pre-trained LightGBM text model (.txt)")
    p.add_argument("--lgbm-threshold", type=float, default=None)
    p.add_argument("--max-malicious", type=int, default=None)
    p.add_argument("--max-benign", type=int, default=None)

    p.add_argument("--D", type=int, default=None)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--F", type=int, default=None)
    p.add_argument("--test-interval", type=int, default=None)
    p.add_argument("--max-ratio", type=float, default=None)

    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--replay-start-size", type=int, default=None)
    p.add_argument("--memory-capacity", type=int, default=None)
    p.add_argument("--target-update-interval", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)

    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None, choices=["cpu", "cuda"])
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    ini = _load_ini(args.config)

    malicious_dir = _get_str(args.train_dir, ini, "Dataset", "malicious_dir")
    if malicious_dir is None:
        malicious_dir = _get_str(args.malicious_dir, ini, "Dataset", "malicious_dir")
    output_dir = _derive_output_dir(
        train_dir=malicious_dir if malicious_dir is not None else "dqeaf",
        configured_output_dir=_get_str(args.output_dir, ini, "Paths", "output_dir"),
    )
    malicious_exclude_dirs = _parse_csv(_get_str(None, ini, "Dataset", "malicious_exclude_dirs", "Benign"))

    classifier_backend = _get_str(args.classifier_backend, ini, "Classifier", "backend", "gbdt")
    classifier_path = _get_str(args.classifier_path, ini, "Classifier", "classifier_path")
    gbdt_model_path = _get_str(
        args.gbdt_model_path,
        ini,
        "Classifier",
        "gbdt_model_path",
        str(ROOT / "LIGHTGBMs" / "gradient_boosting.pkl"),
    )
    gbdt_threshold = _get_float(args.gbdt_threshold, ini, "Classifier", "gbdt_threshold", 0.9)
    lgbm_model_path = _get_str(args.lgbm_model_path, ini, "Classifier", "lgbm_model_path")
    lgbm_threshold = _get_float(args.lgbm_threshold, ini, "Classifier", "lgbm_threshold", 0.8336)

    max_malicious = _get_int(args.max_malicious, ini, "Dataset", "max_malicious", 0)
    max_benign = _get_int(args.max_benign, ini, "Dataset", "max_benign", 0)
    if max_malicious <= 0:
        max_malicious = None
    if max_benign <= 0:
        max_benign = None

    d = _get_int(args.D, ini, "Train", "D", 30000)
    t = _get_int(args.T, ini, "Train", "T", 80)
    f = _get_int(args.F, ini, "Train", "F", 200)
    test_interval = _get_int(args.test_interval, ini, "Train", "test_interval", 1000)
    max_ratio = _get_float(args.max_ratio, ini, "Train", "max_ratio", 7.0)

    gamma = _get_float(args.gamma, ini, "Train", "gamma", 0.99)
    batch_size = _get_int(args.batch_size, ini, "Train", "batch_size", 32)
    replay_start_size = _get_int(args.replay_start_size, ini, "Train", "replay_start_size", 1000)
    memory_capacity = _get_int(args.memory_capacity, ini, "Train", "memory_capacity", 100000)
    target_update_interval = _get_int(args.target_update_interval, ini, "Train", "target_update_interval", 100)
    learning_rate = _get_float(args.learning_rate, ini, "Train", "learning_rate", 1e-3)
    seed = _get_int(args.seed, ini, "Train", "seed", 1337)
    device = _get_str(args.device, ini, "Runtime", "device", "cpu")

    if malicious_dir is None:
        raise RuntimeError("Missing malicious dataset path. Set Dataset.malicious_dir in config.ini or pass --train_dir")

    print(f"[i] Output directory: {output_dir}")
    print("[1/5] Loading corpora...")
    malicious = load_binary_corpus(
        malicious_dir,
        max_malicious,
        exclude_dir_names=malicious_exclude_dirs,
    )

    if len(malicious) < 2:
        raise RuntimeError("Need at least 2 malicious samples")

    train_mal, test_mal = train_test_split_bytes(malicious, test_size=0.2, seed=seed)

    if classifier_backend == "lgbm_ember":
        if not lgbm_model_path:
            raise RuntimeError("Missing lgbm model path. Set Classifier.lgbm_model_path in config.ini or pass --lgbm-model-path")
        print("[2/5] Loading pre-trained LightGBM-EMBER surrogate...")
        classifier = LightGBMEmberClassifier.load_model(
            model_path=lgbm_model_path,
            threshold=lgbm_threshold,
        )
    else:
        if classifier_path:
            print("[2/5] Loading pre-trained independent classifier...")
            classifier = IndependentClassifier.load(classifier_path)
        else:
            if gbdt_model_path is None:
                raise RuntimeError(
                    "Missing pre-trained GBDT model path. Set Classifier.gbdt_model_path in config.ini "
                    "or pass --gbdt-model-path"
                )
            gbdt_model = Path(gbdt_model_path)
            if not gbdt_model.is_absolute():
                gbdt_model = (ROOT / gbdt_model).resolve()
            if not gbdt_model.exists():
                raise RuntimeError(
                    f"Pre-trained GBDT model not found: {gbdt_model}. "
                    "Provide --gbdt-model-path or put model at LIGHTGBMs/gradient_boosting.pkl"
                )
            print(f"[2/5] Loading pre-trained EMBER GBDT surrogate: {gbdt_model}")
            classifier = ScikitLearnEmberClassifier.load_model(gbdt_model, threshold=gbdt_threshold)

    cfg = DQEAFConfig(
        D=d,
        T=t,
        F=f,
        TEST_INTERVAL=test_interval,
        MAX_RATIO=max_ratio,
        gamma=gamma,
        learning_rate=learning_rate,
        minibatch_size=batch_size,
        replay_start_size=replay_start_size,
        memory_capacity=memory_capacity,
        target_update_interval=target_update_interval,
        seed=seed,
        device=device,
    )

    print("[3/5] Initializing trainer...")
    trainer = Trainer(
        config=cfg,
        classifier=classifier,
        train_malware_samples=train_mal,
        test_malware_samples=test_mal,
    )

    print("[4/5] Training DQEAF...")
    history = trainer.train()

    print("[5/5] Saving artifacts...")
    trainer.save(output_dir)

    summary = {
        "episodes_logged": len(history),
        "best_sr": trainer.best_sr,
        "global_step": trainer.global_step,
        "output_dir": output_dir,
        "config": args.config,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
