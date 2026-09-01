"""
GAPGAN - Evaluation / Attack Pipeline
========================================
Paper: "Black-box Adversarial Attacks Against Deep Learning Based Malware
        Binaries Detection with GAN" (ECAI 2020)

Implements the attack process (Figure 1, bottom):
  1. Load trained Generator G.
  2. For each malware sample in the attack set:
     a. Normalize and feed to G → generate adversarial payloads.
     b. Concatenate payloads to original malware → adversarial sample.
     c. Convert back to discrete bytes and save.
  3. Query the black-box detector f on both original and adversarial samples.
  4. Compute Attack Success Rate (ASR, Eq. 7):
       ASR = Σ I(f(x_mal)=-1 ∧ f(x_adv)=1) / Σ I(f(x_mal)=-1)

Reference: Section 4.1 (Eq. 7) & Section 5.
"""

import os
import csv
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_extraction.preprocessor import (
    BinaryDataset, denormalize, normalize, pad_binary, read_binary,
    normalized_to_byte_input,
)
from detector.target_model import MalConv
from gan.generator import Generator
from training.thresholding import apply_dynamic_threshold


def load_checkpoint_state_dict(checkpoint_path: str,
                               device: torch.device) -> dict:
    """Load either a wrapped checkpoint or a raw state dict."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def collect_files(root: str) -> set[str]:
    """Collect all real file paths under a directory."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Directory not found: {root}")

    files = set()
    for base, _, names in os.walk(root):
        for name in names:
            path = os.path.realpath(os.path.join(base, name))
            if os.path.isfile(path):
                files.add(path)
    return files


def assert_disjoint(dir_a: str, dir_b: str) -> None:
    """Ensure two directory trees do not share any files."""
    overlap = collect_files(dir_a) & collect_files(dir_b)
    if overlap:
        sample = sorted(overlap)[0]
        raise ValueError(
            "Detector-train and attack sets must be disjoint; "
            f"found overlap: {sample}"
        )


def require_checkpoint(checkpoint_path: str, label: str) -> None:
    """Fail fast if a required checkpoint is missing."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Missing {label} checkpoint: {checkpoint_path}"
        )


def looks_like_pe(blob: bytes) -> bool:
    """Lightweight PE signature check before exporting as .exe."""
    if len(blob) < 0x40 or blob[:2] != b"MZ":
        return False

    pe_offset = int.from_bytes(blob[0x3C:0x40], byteorder="little")
    if pe_offset < 0 or pe_offset + 4 > len(blob):
        return False

    return blob[pe_offset:pe_offset + 4] == b"PE\x00\x00"


def build_evaded_export_path(original_path: str,
                             malware_dir: str,
                             evaded_dir: str) -> str:
    """Preserve malware-family folder and original executable name."""
    rel_path = os.path.relpath(original_path, malware_dir)
    rel_dir = os.path.dirname(rel_path)
    original_name = os.path.basename(original_path)
    stem, ext = os.path.splitext(original_name)
    export_name = original_name if ext.lower() == ".exe" else f"{stem}.exe"

    export_dir = evaded_dir
    if rel_dir and rel_dir != ".":
        export_dir = os.path.join(evaded_dir, rel_dir)
    os.makedirs(export_dir, exist_ok=True)

    return os.path.join(export_dir, export_name)


def format_optional_score(value: Optional[float]) -> str:
    """Render a score for CSV while allowing score-less export mode."""
    if value is None:
        return ""
    return f"{value:.6f}"





def evaluate_gapgan(
    malware_dir: str,
    benign_dir: str,
    generator_path: str,
    blackbox_path: Optional[str] = None,
    input_length: int = 2_000_000,
    payload_rate: float = 0.025,
    epsilon: float = 0.06,
    output_dir: str = "adversarial_samples",
    evaded_dir: Optional[str] = "evaded_samples",
    device_name: str = "cpu",
    detector_train_root: Optional[str] = None,
    manifest_path: Optional[str] = None,
):
    """
    Run the GAPGAN attack process and compute ASR.

    Parameters
    ----------
    malware_dir : str       Path to malware binaries for attack.
    benign_dir : str        Path to benign binaries (needed for dataset).
    generator_path : str    Path to trained Generator checkpoint.
    blackbox_path : str     Optional path to pre-trained MalConv weights.
    input_length : int      Fixed input size t.
    payload_rate : float    Payload rate.
    epsilon : float         Threshold ε (applied at max iteration level).
    output_dir : str        Directory to save adversarial binaries.
    evaded_dir : str        Directory to save evaded PE files as .exe.
    device_name : str       Device string.
    manifest_path : str     Optional CSV path for per-file save metadata.
    """

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    if evaded_dir is not None:
        os.makedirs(evaded_dir, exist_ok=True)

    if manifest_path is None:
        manifest_path = os.path.join(output_dir, "manifest.csv")
    manifest_dir = os.path.dirname(manifest_path)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)

    payload_length = int(input_length * payload_rate)

    print(f"[EVAL] input_length={input_length}, "
          f"payload_length={payload_length}")

    if detector_train_root is not None:
        assert_disjoint(malware_dir, detector_train_root)
        assert_disjoint(benign_dir, detector_train_root)

    # --- Load models ---
    f = None
    if blackbox_path is not None:
        f = MalConv(input_length=input_length).to(device)
        require_checkpoint(blackbox_path, "trained MalConv")
        f.load_state_dict(load_checkpoint_state_dict(blackbox_path, device))
        f.eval()

    G = Generator(input_length=input_length,
                  payload_length=payload_length).to(device)
    G.load_state_dict(load_checkpoint_state_dict(generator_path, device))
    G.eval()

    if f is None:
        print("[EVAL] Generator loaded. Black-box scoring disabled.")
    else:
        print("[EVAL] Models loaded.")

    # --- Load attack dataset ---
    dataset = BinaryDataset(malware_dir, benign_dir,
                            target_length=input_length)

    # Counters for generation and optional ASR-style reporting.
    total_generated = 0
    pe_exported = 0
    detected_as_malware = 0    # f(x_mal) = -1
    evaded_after_attack = 0    # f(x_mal) = -1 AND f(x_adv) = 1

    manifest_fields = [
        "Index",
        "Original File",
        "Original Path",
        "Adv File",
        "Adv Path",
        "Original Score",
        "Adv Score",
        "Result",
        "Original Length",
        "Adv Length",
        "Exported EXE Path",
    ]

    with open(manifest_path, "w", newline="", encoding="utf-8") as manifest_fout:
        manifest_writer = csv.DictWriter(manifest_fout, fieldnames=manifest_fields)
        manifest_writer.writeheader()

        for idx in range(len(dataset)):
            x, label, original_length = dataset[idx]

            # Only attack malware samples (y = -1)
            if label.item() != -1:
                continue

            x = x.unsqueeze(0).to(device)  # (1, t)
            original_path = os.path.realpath(dataset.samples[idx][0])

            f_orig_score = None
            f_adv_score = None
            result = "Generated"
            if f is not None:
                byte_input_orig = normalized_to_byte_input(x)
                with torch.no_grad():
                    f_orig = f(byte_input_orig)
                f_orig_score = f_orig.item()

            # --- Generate adversarial payload ---
            with torch.no_grad():
                a_adv = G(x)  # (1, payload_length)

                # Paper §3.2: "we abandon the padding zeros for reducing
                # the whole length of payloads in the attack process"
                # Apply threshold at max level (i = T_max)
                a_adv = apply_dynamic_threshold(a_adv, 1, 1, epsilon)

                # Attack-time sample: original malware bytes + generated payload.
                malware_prefix = x[:, :original_length]
                x_adv_file = torch.cat([malware_prefix, a_adv], dim=1)

            # --- Optionally query f for reporting only ---
            if f is not None:
                byte_input_adv = normalized_to_byte_input(x_adv_file)
                with torch.no_grad():
                    f_adv = f(byte_input_adv)
                f_adv_score = f_adv.item()

                if f_orig_score is not None and f_orig_score < 0.5:
                    detected_as_malware += 1
                    if f_adv_score >= 0.5:
                        evaded_after_attack += 1
                        result = "Evaded"
                    else:
                        result = "Detected"
                else:
                    result = "OrigBenign"

            # --- Save adversarial binary ---
            adv_bytes = denormalize(x_adv_file.squeeze(0).cpu().numpy())
            adv_blob = adv_bytes.tobytes()
            out_path = os.path.join(output_dir, f"adv_{idx:06d}.bin")
            with open(out_path, "wb") as fout:
                fout.write(adv_blob)

            exported_exe_path = ""
            if evaded_dir is not None and looks_like_pe(adv_blob):
                exported_out_path = build_evaded_export_path(
                    original_path=original_path,
                    malware_dir=malware_dir,
                    evaded_dir=evaded_dir,
                )
                with open(exported_out_path, "wb") as fout:
                    fout.write(adv_blob)
                exported_exe_path = os.path.realpath(exported_out_path)
                pe_exported += 1

            manifest_writer.writerow({
                "Index": f"{idx:06d}",
                "Original File": os.path.basename(original_path),
                "Original Path": original_path,
                "Adv File": os.path.basename(out_path),
                "Adv Path": os.path.realpath(out_path),
                "Original Score": format_optional_score(f_orig_score),
                "Adv Score": format_optional_score(f_adv_score),
                "Result": result,
                "Original Length": str(original_length),
                "Adv Length": str(len(adv_bytes)),
                "Exported EXE Path": exported_exe_path,
            })
            manifest_fout.flush()

            total_generated += 1

    # --- Compute ASR (Eq. 7) ---
    if detected_as_malware > 0:
        asr = evaded_after_attack / detected_as_malware
    else:
        asr = 0.0

    print("=" * 60)
    print(f"[EVAL] Total adversarial samples:      {total_generated}")
    print(f"[EVAL] PE .exe exports created:        {pe_exported}")
    if f is not None:
        print(f"[EVAL] Detected by f as malware:       {detected_as_malware}")
        print(f"[EVAL] Evaded after attack:            {evaded_after_attack}")
        print(f"[EVAL] Attack Success Rate (ASR):      {asr:.4f} "
              f"({asr*100:.2f}%)")
    print(f"[EVAL] Manifest CSV:                   {manifest_path}")
    if evaded_dir is not None:
        print(f"[EVAL] Exported EXE dir:               {evaded_dir}")
    print("=" * 60)

    return asr


# =========================================================================
# CLI entry point
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GAPGAN Evaluation — Attack Process (ECAI 2020)")

    parser.add_argument("--malware_dir", type=str, required=True,
                        help="Path to malware binaries for attack.")
    parser.add_argument("--benign_dir", type=str, required=True,
                        help="Path to benign binaries.")
    parser.add_argument("--generator_path", type=str, required=True,
                        help="Path to trained Generator checkpoint.")
    parser.add_argument("--blackbox_path", type=str, default=None,
                        help="Optional path to pre-trained MalConv weights "
                             "for reporting/ASR only.")
    parser.add_argument("--input_length", type=int, default=2_000_000,
                        help="Fixed input size t.")
    parser.add_argument("--payload_rate", type=float, default=0.025,
                        help="Payload rate.")
    parser.add_argument("--epsilon", type=float, default=0.06,
                        help="Maximum threshold ε.")
    parser.add_argument("--output_dir", type=str,
                        default="adversarial_samples",
                        help="Directory to save adversarial binaries.")
    parser.add_argument("--evaded_dir", type=str, default="evaded_samples",
                        help="Directory to export generated PE samples as "
                             ".exe under their original family folders.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device.")
    parser.add_argument("--detector_train_root", type=str, default=None,
                        help="Optional detector-training dataset root that "
                             "must be disjoint from the attack dataset.")
    parser.add_argument("--manifest_path", type=str, default=None,
                        help="Optional CSV path for per-file attack metadata. "
                             "Defaults to output_dir/manifest.csv")

    args = parser.parse_args()

    evaluate_gapgan(
        malware_dir=args.malware_dir,
        benign_dir=args.benign_dir,
        generator_path=args.generator_path,
        blackbox_path=args.blackbox_path,
        input_length=args.input_length,
        payload_rate=args.payload_rate,
        epsilon=args.epsilon,
        output_dir=args.output_dir,
        evaded_dir=args.evaded_dir,
        device_name=args.device,
        detector_train_root=args.detector_train_root,
        manifest_path=args.manifest_path,
    )


if __name__ == "__main__":
    main()
