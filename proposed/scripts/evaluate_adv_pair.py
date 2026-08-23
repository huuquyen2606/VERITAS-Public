"""Evaluate one original/candidate PE pair through the Adv-RL Env backend."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.adv_rl_bridge import AdvRLPipelineBridge  


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Adv-RL Env evaluation on one PE pair.")
    parser.add_argument("--ori", required=True, help="Original PE path.")
    parser.add_argument("--adv", required=True, help="Candidate/adversarial PE path.")
    parser.add_argument("--label", required=True, help="Original malware family label.")
    parser.add_argument("--work-dir", default="/tmp", help="Worker staging directory.")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--max-wait-sec", type=int, default=1800)
    parser.add_argument("--skip-detector", action="store_true", help="Skip detector inference.")
    parser.add_argument(
        "--inline-functionality",
        action="store_true",
        help="Run FunctionalityEvaluator in this process instead of waiting for the external Redis listener.",
    )
    parser.add_argument("--refresh-baseline", action="store_true", help="Force re-extract original.")
    parser.add_argument("--output-json", help="Optional path for result JSON.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    bridge = AdvRLPipelineBridge(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
        work_dir=args.work_dir,
        poll_interval=args.poll_interval,
        max_wait_sec=args.max_wait_sec,
        run_detector=not args.skip_detector,
        run_functionality_inline=args.inline_functionality,
    )

    result = bridge.evaluate_pair(
        original_path=args.ori,
        candidate_path=args.adv,
        true_label=args.label,
        refresh_baseline=args.refresh_baseline,
    )

    payload = {
        "original_name": result.original_name,
        "candidate_name": result.candidate_name,
        "true_label": result.true_label,
        "reward": result.reward,
        "integrity_score": result.integrity_score,
        "functionality_score": result.functionality_score,
        "detector_scores": result.detector_scores,
        "evaded": result.evaded,
        "redis_key": result.redis_key,
        "state_shape": list(result.state.shape),
        "static_source": result.static_source,
        "sandbox_context": result.sandbox_context,
    }

    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
