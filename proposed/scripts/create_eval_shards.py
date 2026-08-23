"""Create stratified evaluation shards with absolute symlinks."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PE_SUFFIXES = {".exe", ".dll"}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified test shards for parallel evaluation.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--families", nargs="+", required=True)
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the planned split without writing.")
    return parser.parse_args()


def _family_samples(input_root: Path, family: str) -> list[Path]:
    family_dir = input_root / family
    if not family_dir.is_dir():
        raise FileNotFoundError(f"Missing family directory: {family_dir}")
    samples = sorted(
        path.resolve()
        for path in family_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in PE_SUFFIXES
    )
    if not samples:
        raise RuntimeError(f"No PE samples found for family={family} under {family_dir}")
    return samples


def _place_sample(src: Path, dest: Path, mode: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        if mode == "symlink" and dest.is_symlink() and dest.resolve() == src.resolve():
            return
        if mode == "copy" and dest.is_file() and dest.stat().st_size == src.stat().st_size:
            return
        raise FileExistsError(f"Refusing to overwrite existing shard path: {dest}")

    if mode == "symlink":
        dest.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dest)


def build_manifest(
    *,
    input_root: Path,
    output_root: Path,
    families: list[str],
    shard_count: int,
    mode: str,
    dry_run: bool,
) -> dict[str, Any]:
    if shard_count <= 0:
        raise ValueError("--shards must be positive")

    shard_ids = [f"shard_{idx}" for idx in range(1, shard_count + 1)]
    records: list[dict[str, Any]] = []
    shard_counts: dict[str, Counter[str]] = {shard_id: Counter() for shard_id in shard_ids}
    family_counts: Counter[str] = Counter()

    for family in families:
        samples = _family_samples(input_root, family)
        family_counts[family] = len(samples)
        for idx, src in enumerate(samples):
            shard_id = shard_ids[idx % shard_count]
            dest = output_root / shard_id / family / src.name
            record = {
                "family": family,
                "shard_id": shard_id,
                "source_path": str(src.resolve()),
                "shard_path": str(dest.resolve()),
                "sample_stem": src.stem,
            }
            records.append(record)
            shard_counts[shard_id][family] += 1
            if not dry_run:
                _place_sample(src, dest, mode)

    counts = {
        shard_id: {
            "total": int(sum(counter.values())),
            "by_family": {family: int(counter.get(family, 0)) for family in families},
        }
        for shard_id, counter in shard_counts.items()
    }

    return {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "mode": mode,
        "families": families,
        "shards": shard_ids,
        "total_samples": len(records),
        "family_counts": {family: int(family_counts[family]) for family in families},
        "counts": counts,
        "samples": records,
    }


def main() -> int:
    args = parse_args()
    input_root = _resolve(args.input_root)
    output_root = _resolve(args.output_root)
    manifest_path = _resolve(args.manifest)

    manifest = build_manifest(
        input_root=input_root,
        output_root=output_root,
        families=list(args.families),
        shard_count=int(args.shards),
        mode=str(args.mode),
        dry_run=bool(args.dry_run),
    )

    if not args.dry_run:
        _write_json(manifest_path, manifest)

    print(json.dumps({
        "dry_run": bool(args.dry_run),
        "manifest": str(manifest_path),
        "total_samples": manifest["total_samples"],
        "family_counts": manifest["family_counts"],
        "counts": manifest["counts"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
