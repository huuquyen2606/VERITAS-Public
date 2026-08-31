#!/usr/bin/env python3
"""
Script to prepare malware samples for OBFU-mal.
It scans a dataset directory, ignores benign folders, computes SHA256
for executable samples, and copies them to data/samples.
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from obfumal.utils.sample_preparation import DEFAULT_SAMPLES_DIR, prepare_samples_from_dataset

def main():
    parser = argparse.ArgumentParser(
        description="Copy malware samples into OBFU-mal sample folder with SHA256 filenames."
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Dataset root directory (e.g., ./Adv_agent).",
    )
    parser.add_argument(
        "--dest-dir",
        default=str(DEFAULT_SAMPLES_DIR),
        help="Destination samples directory.",
    )
    parser.add_argument(
        "--exclude-dirs",
        nargs="*",
        default=["Benign"],
        help="Folder names to ignore while scanning.",
    )
    parser.add_argument(
        "--clear-dest",
        action="store_true",
        help="Delete existing hash-named samples in destination before copying.",
    )
    parser.add_argument(
        "--recreate-dest",
        action="store_true",
        help="Delete destination directory and create it again before copying.",
    )
    args = parser.parse_args()

    if args.clear_dest and args.recreate_dest:
        parser.error("--clear-dest and --recreate-dest cannot be used together.")

    summary = prepare_samples_from_dataset(
        source_dir=args.source_dir,
        dest_dir=args.dest_dir,
        excluded_dir_names=args.exclude_dirs,
        clear_destination=args.clear_dest,
        recreate_destination=args.recreate_dest,
    )

    print("\n--- Summary ---")
    print(f"Source dataset: {Path(summary.source_dir)}")
    print(f"Destination: {Path(summary.dest_dir)}")
    print(f"Destination recreated: {summary.recreated_destination}")
    print(f"Cleared existing hash files: {summary.cleared_existing}")
    print(f"Total files scanned: {summary.scanned_files}")
    print(f"Executable candidates: {summary.executable_candidates}")
    print(f"Successfully copied: {summary.copied_files} (unique SHA256 hashes)")
    print(f"Skipped existing (duplicates): {summary.skipped_existing}")
    print(f"Skipped non-executable files: {summary.skipped_non_executable}")
    print(f"Failed: {summary.failed}")

if __name__ == "__main__":
    main()
