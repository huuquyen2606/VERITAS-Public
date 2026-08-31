import argparse
import csv
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from obfumal.obfumal import extract_sequences


def main():
    parser = argparse.ArgumentParser(description="Export action sequences to CSV")
    parser.add_argument("--history", type=str, required=True, help="Path to history JSON file")
    parser.add_argument("--out", type=str, default="artifacts/reports/sequences.csv", help="Output CSV path")
    args = parser.parse_args()

    with open(args.history, "r") as f:
        payload = json.load(f)

    history = payload.get("by_episode", payload) if isinstance(payload, dict) else payload
    rows = extract_sequences(history)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sha256", "evaded", "evaded_sha256", "actions", "original_score"])
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["actions"] = ",".join(row.get("actions", []))
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
