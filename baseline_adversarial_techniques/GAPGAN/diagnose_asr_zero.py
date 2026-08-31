"""
GAPGAN zero-ASR diagnosis utility.

This script compares three scores on the same detected malware samples:
  1. Original malware score f(x_mal)
  2. GAPGAN-crafted score f(x_adv_gapgan)
  3. Random-payload score f(x_adv_random)

If a discriminator checkpoint is provided, it also reports how optimistic the
surrogate D is relative to the black-box detector f on the same crafted sample.
"""

import argparse
import csv
import os
import random
import statistics
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector.target_model import MalConv
from evaluation.evaluate import load_checkpoint_state_dict, require_checkpoint
from feature_extraction.preprocessor import BinaryDataset, normalized_to_byte_input
from gan.discriminator import Discriminator
from gan.generator import Generator
from training.thresholding import apply_dynamic_threshold


def build_random_payload_like(
    reference: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Create a random continuous payload and apply the same threshold rule."""
    random_payload = torch.empty_like(reference).uniform_(-1.0, 1.0)
    return apply_dynamic_threshold(random_payload, 1, 1, epsilon)


def summarize_metric(rows: List[Dict], key: str) -> str:
    values = [row[key] for row in rows if row[key] != ""]
    if not values:
        return "n/a"
    mean = statistics.fmean(values)
    median = statistics.median(values)
    return f"mean={mean:.4f}, median={median:.4f}"


def maybe_write_csv(rows: List[Dict], report_path: str) -> None:
    if not report_path:
        return
    fieldnames = list(rows[0].keys()) if rows else [
        "index",
        "file",
        "orig_len",
        "truncated",
        "f_orig",
        "f_gapgan",
        "f_random",
        "delta_gapgan",
        "delta_random",
        "gap_to_evade_gapgan",
        "gap_to_evade_random",
        "gapgan_evaded",
        "random_evaded",
        "gapgan_active_frac",
        "random_active_frac",
        "gapgan_mean_abs",
        "random_mean_abs",
        "d_gapgan",
        "d_random",
        "gapgan_surrogate_gap",
        "random_surrogate_gap",
    ]
    with open(report_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def diagnose(
    malware_dir: str,
    benign_dir: str,
    generator_path: str,
    blackbox_path: str,
    input_length: int,
    payload_rate: float,
    epsilon: float,
    max_samples: int,
    device_name: str,
    seed: int,
    report_path: str,
    print_limit: int,
    discriminator_path: str,
) -> int:
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    payload_length = int(input_length * payload_rate)

    print(
        f"[DIAG] input_length={input_length}, payload_length={payload_length}, "
        f"device={device}"
    )
    print(f"[DIAG] max_samples={max_samples}, seed={seed}")

    require_checkpoint(blackbox_path, "trained MalConv")
    require_checkpoint(generator_path, "trained Generator")

    f = MalConv(input_length=input_length).to(device)
    f.load_state_dict(load_checkpoint_state_dict(blackbox_path, device))
    f.eval()

    G = Generator(input_length=input_length, payload_length=payload_length).to(device)
    G.load_state_dict(load_checkpoint_state_dict(generator_path, device))
    G.eval()

    D = None
    if discriminator_path:
        require_checkpoint(discriminator_path, "trained Discriminator")
        D = Discriminator(input_length=input_length + payload_length).to(device)
        D.load_state_dict(load_checkpoint_state_dict(discriminator_path, device))
        D.eval()
        print(f"[DIAG] discriminator_path={discriminator_path}")

    dataset = BinaryDataset(malware_dir, benign_dir, target_length=input_length)

    rows: List[Dict] = []
    skipped_not_detected = 0

    for idx in range(len(dataset)):
        x, label, original_length = dataset[idx]
        if label.item() != -1:
            continue

        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            f_orig = f(normalized_to_byte_input(x)).item()

        if f_orig >= 0.5:
            skipped_not_detected += 1
            continue

        with torch.no_grad():
            a_gapgan = G(x)
            a_gapgan = apply_dynamic_threshold(a_gapgan, 1, 1, epsilon)
            malware_prefix = x[:, :original_length]
            x_adv_gapgan = torch.cat([malware_prefix, a_gapgan], dim=1)
            f_gapgan = f(normalized_to_byte_input(x_adv_gapgan)).item()

            a_random = build_random_payload_like(a_gapgan, epsilon)
            x_adv_random = torch.cat([malware_prefix, a_random], dim=1)
            f_random = f(normalized_to_byte_input(x_adv_random)).item()

            d_gapgan = D(x_adv_gapgan).item() if D is not None else ""
            d_random = D(x_adv_random).item() if D is not None else ""

        row = {
            "index": idx,
            "file": os.path.basename(dataset.samples[idx][0]),
            "orig_len": original_length,
            "truncated": int(original_length > input_length),
            "f_orig": round(f_orig, 6),
            "f_gapgan": round(f_gapgan, 6),
            "f_random": round(f_random, 6),
            "delta_gapgan": round(f_gapgan - f_orig, 6),
            "delta_random": round(f_random - f_orig, 6),
            "gap_to_evade_gapgan": round(0.5 - f_gapgan, 6),
            "gap_to_evade_random": round(0.5 - f_random, 6),
            "gapgan_evaded": int(f_gapgan >= 0.5),
            "random_evaded": int(f_random >= 0.5),
            "gapgan_active_frac": round((a_gapgan.abs() > 1e-8).float().mean().item(), 6),
            "random_active_frac": round((a_random.abs() > 1e-8).float().mean().item(), 6),
            "gapgan_mean_abs": round(a_gapgan.abs().mean().item(), 6),
            "random_mean_abs": round(a_random.abs().mean().item(), 6),
            "d_gapgan": round(d_gapgan, 6) if D is not None else "",
            "d_random": round(d_random, 6) if D is not None else "",
            "gapgan_surrogate_gap": round(d_gapgan - f_gapgan, 6) if D is not None else "",
            "random_surrogate_gap": round(d_random - f_random, 6) if D is not None else "",
        }
        rows.append(row)

        if len(rows) >= max_samples:
            break

    print("=" * 72)
    print(f"[DIAG] Malware samples inspected:        {len(rows)}")
    print(f"[DIAG] Skipped (f already benign):       {skipped_not_detected}")
    print(f"[DIAG] Truncated malware in sample set:  {sum(r['truncated'] for r in rows)}")
    print(f"[DIAG] GAPGAN evasion count:             {sum(r['gapgan_evaded'] for r in rows)}")
    print(f"[DIAG] Random evasion count:             {sum(r['random_evaded'] for r in rows)}")
    print(f"[DIAG] f_orig:                           {summarize_metric(rows, 'f_orig')}")
    print(f"[DIAG] f_gapgan:                         {summarize_metric(rows, 'f_gapgan')}")
    print(f"[DIAG] f_random:                         {summarize_metric(rows, 'f_random')}")
    print(f"[DIAG] delta_gapgan:                     {summarize_metric(rows, 'delta_gapgan')}")
    print(f"[DIAG] delta_random:                     {summarize_metric(rows, 'delta_random')}")
    print(f"[DIAG] gap_to_evade_gapgan:              {summarize_metric(rows, 'gap_to_evade_gapgan')}")
    print(f"[DIAG] gapgan_active_frac:               {summarize_metric(rows, 'gapgan_active_frac')}")
    print(f"[DIAG] random_active_frac:               {summarize_metric(rows, 'random_active_frac')}")
    print(f"[DIAG] gapgan_mean_abs:                  {summarize_metric(rows, 'gapgan_mean_abs')}")
    print(f"[DIAG] random_mean_abs:                  {summarize_metric(rows, 'random_mean_abs')}")
    if D is not None:
        print(f"[DIAG] d_gapgan:                         {summarize_metric(rows, 'd_gapgan')}")
        print(f"[DIAG] d_random:                         {summarize_metric(rows, 'd_random')}")
        print(f"[DIAG] gapgan_surrogate_gap:             {summarize_metric(rows, 'gapgan_surrogate_gap')}")
    print("=" * 72)

    if rows:
        print("[DIAG] First samples:")
        for row in rows[:print_limit]:
            line = (
                f"  idx={row['index']:>5} file={row['file']} len={row['orig_len']} "
                f"f_orig={row['f_orig']:.4f} "
                f"f_gapgan={row['f_gapgan']:.4f} "
                f"(d={row['delta_gapgan']:+.4f}) "
                f"f_random={row['f_random']:.4f} "
                f"(d={row['delta_random']:+.4f}) "
                f"gap_act={row['gapgan_active_frac']:.3f}"
            )
            if D is not None:
                line += f" d_gapgan={row['d_gapgan']:.4f}"
            print(line)

        top_improved = sorted(rows, key=lambda row: row["delta_gapgan"], reverse=True)[:5]
        print("[DIAG] Top GAPGAN score increases:")
        for row in top_improved:
            print(
                f"  idx={row['index']:>5} file={row['file']} "
                f"delta_gapgan={row['delta_gapgan']:+.4f} "
                f"f_orig={row['f_orig']:.4f} -> f_gapgan={row['f_gapgan']:.4f}"
            )

        near_evade = sorted(rows, key=lambda row: row["gap_to_evade_gapgan"])[:5]
        print("[DIAG] Top near-evade samples:")
        for row in near_evade:
            line = (
                f"  idx={row['index']:>5} file={row['file']} "
                f"f_gapgan={row['f_gapgan']:.4f} "
                f"gap_to_0.5={row['gap_to_evade_gapgan']:.4f} "
                f"delta={row['delta_gapgan']:+.4f}"
            )
            if D is not None:
                line += (
                    f" d_gapgan={row['d_gapgan']:.4f} "
                    f"(d-f={row['gapgan_surrogate_gap']:+.4f})"
                )
            print(line)

        if D is not None:
            surrogate_overoptimistic = sorted(
                rows, key=lambda row: row["gapgan_surrogate_gap"], reverse=True
            )[:5]
            print("[DIAG] Top surrogate-overoptimistic samples:")
            for row in surrogate_overoptimistic:
                print(
                    f"  idx={row['index']:>5} file={row['file']} "
                    f"d_gapgan={row['d_gapgan']:.4f} "
                    f"f_gapgan={row['f_gapgan']:.4f} "
                    f"(d-f={row['gapgan_surrogate_gap']:+.4f})"
                )

    maybe_write_csv(rows, report_path)
    if report_path:
        print(f"[DIAG] Wrote report: {report_path}")

    if not rows:
        print("[DIAG] No detected malware samples were selected.")
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose why GAPGAN ASR stays at zero on a detected malware subset."
    )
    parser.add_argument("--malware_dir", type=str, required=True)
    parser.add_argument("--benign_dir", type=str, required=True)
    parser.add_argument("--generator_path", type=str, required=True)
    parser.add_argument("--blackbox_path", type=str, required=True)
    parser.add_argument("--input_length", type=int, default=2_000_000)
    parser.add_argument("--payload_rate", type=float, default=0.025)
    parser.add_argument("--epsilon", type=float, default=0.06)
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--print_limit", type=int, default=15)
    parser.add_argument("--report_path", type=str, default="")
    parser.add_argument("--discriminator_path", type=str, default="")
    args = parser.parse_args()

    raise SystemExit(
        diagnose(
            malware_dir=args.malware_dir,
            benign_dir=args.benign_dir,
            generator_path=args.generator_path,
            blackbox_path=args.blackbox_path,
            input_length=args.input_length,
            payload_rate=args.payload_rate,
            epsilon=args.epsilon,
            max_samples=args.max_samples,
            device_name=args.device,
            seed=args.seed,
            report_path=args.report_path,
            print_limit=args.print_limit,
            discriminator_path=args.discriminator_path,
        )
    )


if __name__ == "__main__":
    main()
