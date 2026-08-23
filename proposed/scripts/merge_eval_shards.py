"""Validate and merge parallel evaluation shard outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge parallel VERITAS evaluation shards.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def validate_manifest(manifest: dict[str, Any], shard_ids: list[str]) -> dict[str, Any]:
    samples = manifest.get("samples", [])
    if not isinstance(samples, list) or not samples:
        raise ValueError("Manifest has no samples.")

    expected_shards = set(manifest.get("shards") or [])
    requested_shards = set(shard_ids)
    if expected_shards and requested_shards != expected_shards:
        raise ValueError(f"Requested shards {sorted(requested_shards)} != manifest shards {sorted(expected_shards)}")

    seen_sources: set[str] = set()
    family_counts: Counter[str] = Counter()
    shard_counts: dict[str, Counter[str]] = {shard_id: Counter() for shard_id in shard_ids}

    for record in samples:
        try:
            family = str(record["family"])
            shard_id = str(record["shard_id"])
            source_path = str(Path(record["source_path"]).resolve())
            shard_path = Path(record["shard_path"])
        except KeyError as exc:
            raise ValueError(f"Manifest sample missing required field: {record}") from exc

        if shard_id not in requested_shards:
            raise ValueError(f"Unexpected shard_id in manifest: {shard_id}")
        if source_path in seen_sources:
            raise ValueError(f"Duplicate sample source in manifest: {source_path}")
        seen_sources.add(source_path)

        if not Path(source_path).is_file():
            raise FileNotFoundError(f"Missing source sample: {source_path}")
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard sample/link: {shard_path}")

        family_counts[family] += 1
        shard_counts[shard_id][family] += 1

    manifest_family_counts = manifest.get("family_counts") or {}
    for family, count in sorted(family_counts.items()):
        expected = int(manifest_family_counts.get(family, count))
        if int(count) != expected:
            raise ValueError(f"Family count mismatch for {family}: got {count}, expected {expected}")

    return {
        "samples": len(samples),
        "families": dict(sorted(family_counts.items())),
        "shards": {
            shard_id: {
                "total": int(sum(counter.values())),
                "by_family": dict(sorted(counter.items())),
            }
            for shard_id, counter in sorted(shard_counts.items())
        },
    }


def _merge_reward_distribution(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    total_count = 0
    weighted_sum = 0.0
    min_value = None
    max_value = None

    for metrics in metrics_list:
        reward = metrics.get("reward_distribution") or {}
        count = int(reward.get("count") or 0)
        mean = reward.get("mean")
        if count and mean is not None:
            total_count += count
            weighted_sum += float(mean) * count
        if reward.get("min") is not None:
            min_value = reward["min"] if min_value is None else min(min_value, reward["min"])
        if reward.get("max") is not None:
            max_value = reward["max"] if max_value is None else max(max_value, reward["max"])

    return {
        "count": total_count,
        "min": min_value,
        "max": max_value,
        "mean": float(weighted_sum / total_count) if total_count else None,
    }


def merge_metrics(metrics_list: list[dict[str, Any]], manifest_summary: dict[str, Any], shard_ids: list[str]) -> dict[str, Any]:
    checkpoint_episodes = {metrics.get("checkpoint_episode") for metrics in metrics_list}
    if len(checkpoint_episodes) != 1:
        raise ValueError(f"Shard checkpoint_episode mismatch: {sorted(checkpoint_episodes)}")

    total_samples = sum(int(metrics.get("samples") or 0) for metrics in metrics_list)
    expected_samples = int(manifest_summary["samples"])
    if total_samples != expected_samples:
        raise ValueError(f"Merged sample count mismatch: metrics={total_samples}, manifest={expected_samples}")

    evaded_total = sum(int(metrics.get("evaded") or 0) for metrics in metrics_list)
    action_counter: Counter[str] = Counter()
    family_total: Counter[str] = Counter()
    family_evaded: Counter[str] = Counter()
    steps_weighted_sum = 0.0
    steps_weight = 0

    for metrics in metrics_list:
        action_counter.update(metrics.get("action_usage_freq") or {})
        for family, payload in (metrics.get("per_family_evasion") or {}).items():
            family_total[family] += int(payload.get("samples") or 0)
            family_evaded[family] += int(payload.get("evaded") or 0)

        mean_steps = metrics.get("mean_steps_to_evade")
        evaded = int(metrics.get("evaded") or 0)
        if mean_steps is not None and evaded:
            steps_weighted_sum += float(mean_steps) * evaded
            steps_weight += evaded

    per_family = {
        family: {
            "samples": int(family_total[family]),
            "evaded": int(family_evaded[family]),
            "evasion_rate": float(family_evaded[family] / family_total[family]) if family_total[family] else 0.0,
        }
        for family in sorted(family_total)
    }

    return {
        "checkpoint": metrics_list[0].get("checkpoint"),
        "checkpoint_episode": metrics_list[0].get("checkpoint_episode"),
        "samples": int(total_samples),
        "evaded": int(evaded_total),
        "evasion_rate": float(evaded_total / total_samples) if total_samples else 0.0,
        "mean_steps_to_evade": float(steps_weighted_sum / steps_weight) if steps_weight else None,
        "per_family_evasion": per_family,
        "action_usage_freq": dict(sorted(action_counter.items())),
        "reward_distribution": _merge_reward_distribution(metrics_list),
        "shards": shard_ids,
        "manifest_summary": manifest_summary,
        "elapsed_sec": round(sum(float(metrics.get("elapsed_sec") or 0.0) for metrics in metrics_list), 3),
    }


def read_traces(root: Path, shard_ids: list[str]) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    seen_samples: set[tuple[str | None, str]] = set()

    for shard_id in shard_ids:
        trace_dir = root / "traces" / shard_id
        if not trace_dir.is_dir():
            raise FileNotFoundError(f"Missing trace directory: {trace_dir}")
        for path in sorted(trace_dir.glob("*.json")):
            payload = _read_json(path)
            key = (payload.get("run_id"), str(payload.get("sample")))
            if key in seen_samples:
                raise ValueError(f"Duplicate trace sample: {key}")
            seen_samples.add(key)
            traces.append(payload)

    return traces


def main() -> int:
    args = parse_args()
    root = _resolve(args.root)
    manifest = _read_json(_resolve(args.manifest))
    shard_ids = list(args.shards)
    manifest_summary = validate_manifest(manifest, shard_ids)

    if args.validate_only:
        print(json.dumps({"valid": True, "manifest_summary": manifest_summary}, indent=2, sort_keys=True))
        return 0

    metrics_list = []
    for shard_id in shard_ids:
        metrics_path = root / "metrics" / f"{shard_id}.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing shard metrics: {metrics_path}")
        metrics = _read_json(metrics_path)
        if metrics.get("run_id") != shard_id:
            raise ValueError(f"Shard metrics run_id mismatch in {metrics_path}: {metrics.get('run_id')}")
        metrics_list.append(metrics)

    merged_metrics = merge_metrics(metrics_list, manifest_summary, shard_ids)
    traces = read_traces(root, shard_ids)
    if len(traces) != merged_metrics["samples"]:
        raise ValueError(f"Trace count mismatch: traces={len(traces)}, samples={merged_metrics['samples']}")

    _write_json(root / "metrics.json", merged_metrics)
    _write_json(root / "traces.json", {"traces": traces})
    print(json.dumps({
        "metrics": str(root / "metrics.json"),
        "traces": str(root / "traces.json"),
        "samples": merged_metrics["samples"],
        "evaded": merged_metrics["evaded"],
        "evasion_rate": merged_metrics["evasion_rate"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
