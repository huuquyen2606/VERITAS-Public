"""Build a 2717D VERITAS state vector from an Adv-RL Redis hash or JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.state_builder import AdvStateBuilder, STATE_DIM  


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VERITAS state from Redis/JSON Env data.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--redis-key", help="Redis hash key, e.g. episode:sample.exe:data")
    source.add_argument("--json-in", help="JSON object containing Redis-like fields.")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--label", help="Original malware family label.")
    parser.add_argument("--family-map-json", help="Optional JSON mapping label -> 0..4.")
    parser.add_argument("--pe-path", help="Optional PE path for EMBER static features.")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--actions", default="", help="Comma-separated action IDs already used.")
    parser.add_argument("--output-npy", help="Optional path to write state .npy.")
    parser.add_argument("--context-json", help="Optional path to write sandbox context JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_input(args)
    family_map = load_family_map(args.family_map_json)
    actions = [int(x) for x in args.actions.split(",") if x.strip()]

    builder = AdvStateBuilder(family_to_index=family_map, max_steps=args.max_steps)
    result = builder.build_from_redis_hash(
        data,
        step_count=args.step,
        previous_actions=actions,
        true_label=args.label,
        pe_path=args.pe_path,
    )

    print(f"state_shape={result.state.shape}")
    print(f"state_dim_ok={result.state.shape == (STATE_DIM,)}")
    print(f"static_source={result.static_source}")
    print(f"static_shape={result.static.shape}")
    print(f"dynamic_shape={result.dynamic.shape}")
    print(f"code_shape={result.code.shape}")
    print(f"env_shape={result.env.shape}")
    print(f"sandbox_valid={result.sandbox_context.get('sandbox_valid')}")
    print(f"target_api_count={len(result.sandbox_context.get('target_api_set', []))}")

    if args.output_npy:
        out_path = Path(args.output_npy)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, result.state)

    if args.context_json:
        ctx_path = Path(args.context_json)
        ctx_path.parent.mkdir(parents=True, exist_ok=True)
        ctx_path.write_text(json.dumps(result.sandbox_context, indent=2, sort_keys=True), encoding="utf-8")

    return 0


def load_input(args: argparse.Namespace) -> dict:
    if args.json_in:
        return json.loads(Path(args.json_in).read_text(encoding="utf-8"))

    import redis

    client = redis.Redis(
        host=args.redis_host,
        port=args.redis_port,
        db=args.redis_db,
        decode_responses=True,
    )
    data = client.hgetall(args.redis_key)
    if not data:
        raise SystemExit(f"Redis key not found or empty: {args.redis_key}")
    return data


def load_family_map(path: str | None) -> dict[str, int]:
    if not path:
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(key): int(value) for key, value in raw.items()}


if __name__ == "__main__":
    raise SystemExit(main())
