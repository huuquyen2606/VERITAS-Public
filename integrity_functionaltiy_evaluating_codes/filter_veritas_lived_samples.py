#!/usr/bin/env python3
"""
filter_fiam_lived_samples.py

Filters the FIAM adversarial dataset (by default: vm_resulst/fiam_new/fiam_full.npz)
using the comprehensive report (by default: vm_resulst/fiam_new/comprehensive_detailed_reports.json)
so that the output dataset contains ONLY qualified/lived samples (where both integrity and functionality pass, i.e. is_alive == True and is_functional == True).
"""

import os
import sys
import json
import argparse
import numpy as np
import re


def ensure_str(val):
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="ignore").strip()
    elif hasattr(val, "item"):
        item_val = val.item()
        if isinstance(item_val, bytes):
            return item_val.decode("utf-8", errors="ignore").strip()
        return str(item_val).strip()
    return str(val).strip()


def get_base_name(filename):
    filename = ensure_str(filename)
    match = re.search(r"([a-fA-F0-9]{32,64})", filename)
    if match:
        return match.group(1)
    return filename.split(".exe")[0] + ".exe"


def main():
    parser = argparse.ArgumentParser(
        description="Filter FIAM dataset to keep only qualified (integrity + functional) samples based on comprehensive_detailed_reports.json."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="vm_resulst/fiam_new/fiam_full.npz",
        help="Path to the input NPZ dataset (default: vm_resulst/fiam_new/fiam_full.npz)",
    )
    parser.add_argument(
        "--report",
        "-r",
        default="vm_resulst/fiam_new/comprehensive_detailed_reports.json",
        help="Path to the detailed report JSON (default: vm_resulst/fiam_new/comprehensive_detailed_reports.json)",
    )
    parser.add_argument(
        "--dataset-key",
        "-k",
        default="fiam",
        help="Key inside the report JSON for the fiam dataset (default: fiam)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="vm_resulst/fiam_new/fiam_1500_filtered.npz",
        help="Path to save the filtered NPZ dataset (default: vm_resulst/fiam_new/fiam_1500_filtered.npz)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrites the input file directly (ignores --output)",
    )

    args = parser.parse_args()

    input_path = args.input
    report_path = args.report
    dataset_key = args.dataset_key
    output_path = input_path if args.overwrite else args.output

    if not os.path.exists(input_path):
        print(f"[!] Error: Input dataset not found at '{input_path}'")
        sys.exit(1)

    if not os.path.exists(report_path):
        print(f"[!] Error: Report JSON not found at '{report_path}'")
        sys.exit(1)

    print(f"[*] Loading detailed report from: {report_path}")
    with open(report_path, "r", encoding="utf-8") as f:
        full_report = json.load(f)

    if dataset_key not in full_report:
        # Try case-insensitive lookup
        found = False
        for k in full_report.keys():
            if k.lower() == dataset_key.lower():
                dataset_key = k
                found = True
                break
        if not found:
            print(
                f"[!] Error: Could not find dataset key '{dataset_key}' in {report_path}. Available keys: {list(full_report.keys())}"
            )
            sys.exit(1)

    report_data = full_report[dataset_key]
    total_report_items = len(report_data)
    lived_report_count = sum(
        1
        for v in report_data.values()
        if v.get("is_functional", False) and v.get("is_alive", False)
    )

    print(
        f"[*] Found report for key '{dataset_key}': {total_report_items} total samples in report, {lived_report_count} qualified/lived (is_alive=True and is_functional=True)."
    )

    # Create fallback mapping by base name
    report_by_base = {get_base_name(k): v for k, v in report_data.items()}

    print(f"[*] Loading NPZ dataset from: {input_path}")
    npz_data = np.load(input_path, allow_pickle=True)

    if "name" not in npz_data.files:
        print("[!] Error: 'name' array not found in NPZ archive.")
        sys.exit(1)

    names = npz_data["name"]
    total_samples = len(names)
    print(
        f"    Loaded {total_samples} samples across feature arrays: {list(npz_data.files)}"
    )

    keep_indices = []
    for idx, name_val in enumerate(names):
        name_str = ensure_str(name_val)
        info = report_data.get(name_str) or report_by_base.get(get_base_name(name_str))

        if info and info.get("is_functional", False) and info.get("is_alive", False):
            keep_indices.append(idx)

    print(f"[*] Filtering dataset...")
    print(
        f"    Kept {len(keep_indices)} qualified samples out of {total_samples} ({len(keep_indices)/total_samples*100:.2f}%)."
    )

    # Build dictionary of filtered arrays
    filtered_data = {}
    for key in npz_data.files:
        arr = npz_data[key]
        if getattr(arr, "shape", (0,))[0] == total_samples or len(arr) == total_samples:
            filtered_data[key] = arr[keep_indices]
        else:
            print(
                f"    [!] Warning: Array '{key}' has length {len(arr)} which does not match total_samples ({total_samples}). Keeping unmodified."
            )
            filtered_data[key] = arr

    # Ensure parent directory for output exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"[*] Saving filtered dataset to: {output_path}")
    np.savez_compressed(output_path, **filtered_data)
    print("[+] Done! Filtered FIAM dataset successfully created.")


if __name__ == "__main__":
    main()
