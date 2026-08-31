#!/usr/bin/env python
from __future__ import annotations

import argparse
import faulthandler
import os
import sys
import time
from pathlib import Path

faulthandler.enable()

ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / "dqeaf-venv" / "bin" / "python3"
if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != (ROOT / "dqeaf-venv").resolve():
    print(f"[*] Redirecting execution to {VENV_PYTHON} to use project dependencies...")
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from dqeaf.classifier import load_classifier_artifact
from dqeaf.env import MalwareEnv
from dqeaf.network import DQEAFNet


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _parse_csv(raw: str | None) -> set[str]:
    if raw is None:
        return set()
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def scan_binary_samples(
    base_dir: str | Path,
    exclude_dir_names: set[str] | None = None,
    max_files: int | None = None,
) -> tuple[list[bytes], list[Path], Path]:
    root = Path(base_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")

    excluded = {x.lower() for x in (exclude_dir_names or set())}
    files: list[Path] = []
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if excluded:
            rel_parts = file_path.relative_to(root).parts[:-1]
            if any(part.lower() in excluded for part in rel_parts):
                continue
        files.append(file_path)
    files.sort()

    if max_files is not None and max_files > 0:
        files = files[:max_files]

    samples: list[bytes] = []
    sample_paths: list[Path] = []
    for file_path in files:
        try:
            samples.append(file_path.read_bytes())
            sample_paths.append(file_path)
        except Exception as exc:
            print(f"[-] Failed to read file {file_path}: {exc}")
    return samples, sample_paths, root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 2: use the trained agent to generate Adversarial Examples"
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Directory containing the trained model artifacts (q_target.pt + classifier_meta.json)",
    )
    parser.add_argument(
        "--test_dir",
        required=True,
        help="Root directory of test malware samples (recursive scan over sub-folders)",
    )
    parser.add_argument(
        "--exclude_dirs",
        default="Benign",
        help="Comma-separated sub-folder names to skip. Default: Benign",
    )
    parser.add_argument("--max_turn", type=int, default=10, help="Maximum turns allowed per sample")
    parser.add_argument("--max_files", type=int, default=0, help="Limit the number of test files (0 = no limit)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--output_dir",
        default="Evaded_Malware",
        help="Output directory for AEs. Preserves the relative family sub-folder structure from test_dir",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_start = time.time()
    excluded = _parse_csv(args.exclude_dirs)

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")

    q_target_path = model_dir / "q_target.pt"
    if not q_target_path.exists():
        raise FileNotFoundError(f"Model file not found: {q_target_path}")

    print(f"[*] Loading classifier artifact from: {model_dir}")
    classifier = load_classifier_artifact(model_dir)

    print(f"[*] Loading q_target from: {q_target_path}")
    q_target = DQEAFNet(input_dim=513, action_dim=4)
    q_target.load_state_dict(torch.load(q_target_path, map_location=args.device))
    q_target.to(args.device)
    q_target.eval()

    max_files = args.max_files if args.max_files > 0 else None
    print(f"[*] Loading test malware from: {args.test_dir}")
    test_bytes, test_paths, test_root = scan_binary_samples(
        args.test_dir,
        exclude_dir_names=excluded,
        max_files=max_files,
    )
    print(f"[+] Test samples: {len(test_bytes)}")
    if not test_bytes:
        raise RuntimeError(f"No valid test samples found in {args.test_dir}")

    env = MalwareEnv(test_bytes, classifier, max_turn=args.max_turn, seed=123)

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    success_count = 0
    total_samples = len(test_bytes)
    with torch.no_grad():
        for i, original_bytes in enumerate(test_bytes):
            relative_path = test_paths[i].relative_to(test_root)
            family_name = relative_path.parent.as_posix() if relative_path.parent != Path(".") else "(root)"
            sample_start = time.time()
            print(
                f"[>] [{i + 1}/{total_samples}] Processing {relative_path.as_posix()} | "
                f"Family: {family_name} | Current successes: {success_count}"
            )
            obs = env.reset_with_sample(original_bytes)
            sample_success = False
            last_turn = 0

            for turn in range(1, args.max_turn + 1):
                last_turn = turn
                s_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(args.device)
                action = int(torch.argmax(q_target(s_tensor), dim=1).item())
                obs, reward, terminated, truncated, _info = env.step(action)

                if reward > 0:
                    sample_success = True
                    success_count += 1
                    out_dir = output_root / relative_path.parent
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"evaded_{relative_path.name}"
                    with out_path.open("wb") as f_out:
                        f_out.write(env._current_sample)
                    print(
                        f"[+] [{i + 1}/{total_samples}] Evasion succeeded after {turn} turns | "
                        f"Saved to: {out_path} | Successes: {success_count}/{i + 1} | "
                        f"Sample time: {_format_duration(time.time() - sample_start)}"
                    )
                    break

                if terminated or truncated:
                    break

            if not sample_success:
                print(
                    f"[-] [{i + 1}/{total_samples}] Evasion failed after {last_turn} turns | "
                    f"Successes: {success_count}/{i + 1} | "
                    f"Sample time: {_format_duration(time.time() - sample_start)}"
                )

    print(f"[+] Completed! Successfully generated {success_count}/{len(test_bytes)} Adversarial Examples.")
    print(f"[+] AEs saved to: {output_root}")
    print(f"[+] Total runtime: {_format_duration(time.time() - run_start)}")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
