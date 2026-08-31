import argparse
import copy
import os
import sys
import importlib
import shutil
from pathlib import Path
from typing import Dict, Any, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from obfumal.utils.config import apply_ini_overrides
from obfumal.utils.preflight import detector_details, probe_detector_on_samples
from obfumal.obfumal import LightGBMDetector, SampleStore, default_action_set


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def check_imports(modules: List[str]) -> List[str]:
    missing = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception:
            missing.append(name)
    return missing


def tool_status(tool: str):
    path = shutil.which(tool)
    if path:
        return True, path
    if os.path.exists(tool):
        return True, os.path.abspath(tool)
    # Local fallback for cloned darkarmour repo
    if tool == "darkarmour":
        candidate = os.path.join("darkarmour-master", "darkarmour-master", "darkarmour.py")
        if os.path.exists(candidate):
            return True, os.path.abspath(candidate)
    return False, None


def check_samples(sample_dir: str) -> bool:
    if not os.path.exists(sample_dir):
        return False
    files = [f for f in os.listdir(sample_dir) if os.path.isfile(os.path.join(sample_dir, f))]
    return len(files) > 0


def main():
    parser = argparse.ArgumentParser(description="OBFU-mal readiness check")
    parser.add_argument("--config", type=str, default="configs/evaluation.yaml", help="Path to config file")
    parser.add_argument("--probe-size", type=int, default=64, help="Detector probe sample size")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for probing")
    parser.add_argument("--show-lief-log", action="store_true", help="Show LIEF parser warnings during detector probe")
    args = parser.parse_args()

    if not args.show_lief_log:
        try:
            import lief

            if hasattr(lief, "logging") and hasattr(lief.logging, "disable"):
                lief.logging.disable()
        except Exception:
            pass

    raw_config = load_config(args.config)
    config = apply_ini_overrides(copy.deepcopy(raw_config))

    env_cfg = config.get("env", {})
    det_cfg = config.get("detector", {})
    raw_env_cfg = raw_config.get("env", {})
    raw_det_cfg = raw_config.get("detector", {})

    sample_dir = env_cfg.get("sample_dir", "data/samples")
    output_dir = env_cfg.get("output_dir", "artifacts/evaded/blackbox")
    model_path = det_cfg.get("model_path", "LIGHTGBMs/result.txt")
    threshold = det_cfg.get("threshold", 0.9)
    feature_backend = det_cfg.get("feature_backend", "auto")
    skip_benign = bool(env_cfg.get("skip_benign", True))

    print("== Ready Check ==")
    print(f"Python: {sys.executable}")
    print(f"Config: {args.config}")
    print(f"Sample dir: {sample_dir}")
    print(f"Output dir: {output_dir}")
    print(f"LightGBM model: {model_path}")
    print(f"Threshold: {threshold}")
    print(f"Feature backend: {feature_backend}")
    if raw_env_cfg.get("sample_dir") != sample_dir:
        print(f"Config override: env.sample_dir {raw_env_cfg.get('sample_dir')} -> {sample_dir}")
    if raw_det_cfg.get("model_path") != model_path:
        print(f"Config override: detector.model_path {raw_det_cfg.get('model_path')} -> {model_path}")

    # Pass 1: basic files
    ok_sample = check_samples(sample_dir)
    ok_model = os.path.exists(model_path)
    print(f"Samples present: {ok_sample}")
    print(f"Model file exists: {ok_model}")

    # Pass 2: python deps
    required = ["gymnasium", "numpy", "torch", "lief", "lightgbm", "yaml", "tqdm", "sklearn"]
    missing_py = check_imports(required)
    print(f"Missing python packages: {missing_py if missing_py else 'None'}")

    # Pass 3: external tools
    upx_ok, upx_path = tool_status("upx")
    dark_ok, dark_path = tool_status("darkarmour")
    print(f"UPX: {'FOUND' if upx_ok else 'MISSING'}" + (f" ({upx_path})" if upx_path else ""))
    print(f"Darkarmour: {'FOUND' if dark_ok else 'MISSING'}" + (f" ({dark_path})" if dark_path else ""))
    if not upx_ok:
        print("  - Impact: UPXPack action will no-op.")
    if not dark_ok:
        print("  - Impact: DarkarmourXOR_EL1/2/3 actions will no-op.")
    if dark_ok and dark_path and dark_path.lower().endswith(".py"):
        gcc_ok, gcc_path = tool_status("i686-w64-mingw32-gcc")
        gpp_ok, gpp_path = tool_status("i686-w64-mingw32-g++")
        print(f"Darkarmour compiler i686-w64-mingw32-gcc: {'FOUND' if gcc_ok else 'MISSING'}" + (f" ({gcc_path})" if gcc_path else ""))
        print(f"Darkarmour compiler i686-w64-mingw32-g++: {'FOUND' if gpp_ok else 'MISSING'}" + (f" ({gpp_path})" if gpp_path else ""))
        if not (gcc_ok and gpp_ok):
            print("  - Impact: darkarmour.py may fail to build output binaries.")

    # Action-space check
    actions = default_action_set(
        include_obfuscation=env_cfg.get("include_obfuscation", True),
        upx_path=env_cfg.get("upx_path", "upx"),
        darkarmour_path=env_cfg.get("darkarmour_path", "darkarmour"),
    )
    action_names = [a.name for a in actions]
    required_xor = {"DarkarmourXOR_EL1", "DarkarmourXOR_EL2", "DarkarmourXOR_EL3"}
    has_xor_levels = required_xor.issubset(set(action_names))
    print(f"Action-space size: {len(action_names)}")
    print(f"XOR EL1/EL2/EL3 present: {has_xor_levels}")
    if not has_xor_levels:
        print("  - Impact: extended obfuscation action space is incomplete.")

    # Feature extractor and detector probe
    detector_ok = True
    try:
        from obfumal.env.state import FeatureExtractor

        fx = FeatureExtractor(backend=feature_backend)
        print(f"Feature extractor: OK (source={fx.source})")

        if det_cfg.get("type", "lightgbm") == "lightgbm" and ok_model and ok_sample:
            detector = LightGBMDetector(
                model_path=model_path,
                threshold=float(threshold),
                feature_backend=feature_backend,
            )
            details = detector_details(detector)
            print(
                f"Detector details: name={details['name']} model={details['model_path']} "
                f"threshold={details['threshold']} feature_source={details['feature_source']}"
            )
            store = SampleStore(sample_dir)
            stats = probe_detector_on_samples(
                detector=detector,
                sample_store=store,
                sha256_list=store.list(),
                sample_size=max(1, args.probe_size),
                seed=args.seed,
            )
            print(
                f"Detector probe: sampled={stats['sampled']} malicious={stats['malicious_count']} "
                f"errors={stats['errors']} score_min={stats['score_min']} "
                f"score_max={stats['score_max']} score_mean={stats['score_mean']}"
            )
            if skip_benign and stats["sampled"] > 0 and stats["malicious_count"] == 0:
                detector_ok = False
                print("  - Impact: skip_benign=True may loop/reset forever (no malware predicted).")
    except Exception as e:
        detector_ok = False
        print(f"Feature extractor: FAIL ({e})")

    print("== Done ==")
    if missing_py or not ok_model or not ok_sample or not detector_ok or not has_xor_levels:
        sys.exit(1)


if __name__ == "__main__":
    main()
