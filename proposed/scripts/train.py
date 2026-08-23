"""Train the VERITAS malware-evasion agent end to end."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.veritas_agent import VERITASAgent  
from agent.per_buffer import PrioritizedReplayBuffer  
from env.malware_env import MalwareEnv, OBS_DIM  
from scripts.rl_entrypoint_utils import (  
    choose_device,
    discover_samples,
    env_kwargs_from_config,
    limit_samples_per_family,
    load_checkpoint,
    load_yaml_config,
    make_agent_kwargs,
    make_buffer_kwargs,
    optional_path,
    resolve_path,
    save_checkpoint,
    set_global_seed,
    summarize_action_usage,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VERITAS-Proposal agent.")
    parser.add_argument("--config", default="configs/train.yaml", help="YAML config path.")
    parser.add_argument("--episodes", type=int, default=None, help="Override train.episodes.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of episodes for smoke runs.")
    parser.add_argument("--samples-per-family", type=int, default=None, help="Use at most N samples from each family.")
    parser.add_argument("--device", default=None, help="Override agent.device.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume weights/optimizer from.")
    parser.add_argument("--work-dir", default="/tmp", help="Shared staging directory used by Adv-RL workers.")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--poll-interval", type=float, default=None)
    parser.add_argument("--max-wait-sec", type=int, default=None)
    parser.add_argument("--skip-detector", action="store_true", help="Skip MAttnHealth inference.")
    parser.add_argument("--inline-functionality", action="store_true", help="Run functionality evaluator in-process.")
    parser.add_argument("--prepare-cfg-calls", action=argparse.BooleanOptionalAction, default=True, help="angr call-site prep for actions 17/18 (default: ON).")
    parser.add_argument("--evaluate-reset", action="store_true", help="Extract original sample during reset.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first episode error.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml_config(args.config)

    train_cfg = cfg.get("train", {})
    agent_cfg = cfg.get("agent", {})
    seed = int(train_cfg.get("seed", 1337))
    set_global_seed(seed)

    requested_device = args.device or agent_cfg.get("device", "cpu")
    device = choose_device(requested_device)

    samples = discover_samples(cfg["dataset"])
    samples = limit_samples_per_family(samples, args.samples_per_family)
    if train_cfg.get("shuffle_samples", True):
        random.shuffle(samples)

    total_episodes = int(args.episodes or train_cfg.get("episodes", len(samples)))
    if args.limit is not None:
        total_episodes = min(total_episodes, int(args.limit))
    if total_episodes <= 0:
        print(
            "[+] No training episodes requested; "
            f"config OK, samples={len(samples)}, device={device}."
        )
        return 0

    checkpoint_dir = resolve_path(train_cfg.get("checkpoint_dir", "models/checkpoints/run_001"))
    log_dir = resolve_path(train_cfg.get("log_dir", "logs/runs/run_001"))
    evasive_dir = optional_path(train_cfg.get("evasive_folder"))
    checkpoint_every = int(train_cfg.get("checkpoint_every") or 0)
    log_every = int(train_cfg.get("log_every") or 1)
    batch_size = int(train_cfg.get("batch_size", 32))
    warmup_steps = int(train_cfg.get("warmup_steps", 500))
    target_sync_steps = int(agent_cfg.get("target_sync_steps", 1000))

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if evasive_dir is not None:
        evasive_dir.mkdir(parents=True, exist_ok=True)

    agent = VERITASAgent(**make_agent_kwargs(cfg, device))
    buffer = PrioritizedReplayBuffer(**make_buffer_kwargs(cfg, OBS_DIM))

    global_step = 0
    start_episode = 1
    if args.resume:
        resume_path = resolve_path(args.resume, must_exist=True)
        checkpoint = load_checkpoint(resume_path, agent, device, load_optimizer=True)
        global_step = int(checkpoint.get("global_step", 0))
        start_episode = int(checkpoint.get("episode", 0)) + 1
        print(f"[+] Resumed checkpoint {resume_path} at episode={start_episode} step={global_step}")

    env = MalwareEnv(**env_kwargs_from_config(cfg, args))

    history_path = log_dir / "train_history.jsonl"
    summary_path = log_dir / "train_summary.json"
    latest_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best.pt"

    episode_rewards: list[float] = []
    rolling_rewards: deque[float] = deque(maxlen=max(1, log_every))
    all_actions: list[int] = []
    evasion_count = 0
    best_score = (-1.0, float("-inf"))
    started_at = time.time()

    print(
        "[+] Training VERITAS: "
        f"samples={len(samples)} episodes={total_episodes} device={device} "
        f"buffer_capacity={buffer.capacity} obs_dim={OBS_DIM}"
    )

    sample_cursor = 0
    for episode in range(start_episode, total_episodes + 1):
        if sample_cursor >= len(samples):
            sample_cursor = 0
            if train_cfg.get("shuffle_samples", True):
                random.shuffle(samples)
        sample = samples[sample_cursor]
        sample_cursor += 1

        sample_path = Path(sample["path"])
        label = str(sample["label"])
        total_reward = 0.0
        trace: list[dict[str, Any]] = []
        evaded = False
        episode_error: str | None = None
        loss_value: float | None = None

        try:
            state, _ = env.reset(
                options={
                    "malware_path": str(sample_path),
                    "true_label": label,
                }
            )

            for _ in range(int(cfg.get("env", {}).get("max_steps", 5))):
                action = agent.select_action(state, eval_mode=False)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)

                buffer.push(state, action, reward, next_state, float(done))
                all_actions.append(int(action))
                total_reward += float(reward)
                global_step += 1

                if len(buffer) >= batch_size and global_step >= warmup_steps:
                    batch = buffer.sample(batch_size)
                    states, actions, rewards, next_states, dones, weights, tree_idxs = batch
                    loss_value, td_errors = agent.train_step(
                        (states, actions, rewards, next_states, dones, weights)
                    )
                    buffer.update_priorities(tree_idxs, td_errors)

                if target_sync_steps > 0 and global_step % target_sync_steps == 0:
                    agent.sync_target_network()

                trace.append(
                    {
                        "step": int(info.get("step", len(trace) + 1)),
                        "action": int(action),
                        "agent_action": info.get("agent_action", int(action)),
                        "action_name": info.get("action_name"),
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "evaded": bool(info.get("evaded", False)),
                        "committed": bool(info.get("committed", False)),
                        "rolled_back": bool(info.get("rolled_back", False)),
                        "reward_source": info.get("reward_source"),
                        "integrity_score": info.get("integrity_score"),
                        "functionality_score": info.get("functionality_score"),
                    }
                )

                state = next_state
                if done:
                    evaded = bool(info.get("evaded", False))
                    break

            if evaded:
                evasion_count += 1
                if evasive_dir is not None:
                    out_name = f"{label}__{sample_path.stem}__ep{episode:06d}.exe"
                    (evasive_dir / out_name).write_bytes(env.pe_bytes)

        except Exception as exc:
            episode_error = str(exc)
            if args.fail_fast:
                raise

        episode_rewards.append(total_reward)
        rolling_rewards.append(total_reward)
        evasion_rate = evasion_count / max(1, episode)
        mean_reward = sum(episode_rewards) / len(episode_rewards)
        rolling_reward = sum(rolling_rewards) / len(rolling_rewards)

        record = {
            "episode": episode,
            "global_step": global_step,
            "sample": str(sample_path.relative_to(PROJECT_ROOT) if sample_path.is_relative_to(PROJECT_ROOT) else sample_path),
            "label": label,
            "total_reward": total_reward,
            "steps": len(trace),
            "evaded": evaded,
            "loss": loss_value,
            "buffer_size": len(buffer),
            "evasion_rate": evasion_rate,
            "mean_reward": mean_reward,
            "rolling_reward": rolling_reward,
            "error": episode_error,
            "trace": trace,
        }
        write_jsonl(history_path, record)

        stats = {
            "episodes": episode,
            "global_step": global_step,
            "evasion_count": evasion_count,
            "evasion_rate": evasion_rate,
            "mean_reward": mean_reward,
            "rolling_reward": rolling_reward,
            "buffer_size": len(buffer),
            "action_usage": summarize_action_usage(all_actions),
            "elapsed_sec": round(time.time() - started_at, 3),
        }

        if episode % log_every == 0 or episode == 1 or episode == total_episodes:
            print(
                f"[TRAIN] ep={episode}/{total_episodes} step={global_step} "
                f"label={label} reward={total_reward:.2f} evaded={evaded} "
                f"ER={evasion_rate:.3f} rollR={rolling_reward:.2f} "
                f"buf={len(buffer)} loss={loss_value}"
            )

        current_score = (evasion_rate, mean_reward)
        if current_score > best_score:
            best_score = current_score
            save_checkpoint(best_path, agent=agent, cfg=cfg, episode=episode, global_step=global_step, stats=stats)

        save_checkpoint(latest_path, agent=agent, cfg=cfg, episode=episode, global_step=global_step, stats=stats)
        if checkpoint_every > 0 and episode % checkpoint_every == 0:
            save_checkpoint(
                checkpoint_dir / f"episode_{episode:06d}.pt",
                agent=agent,
                cfg=cfg,
                episode=episode,
                global_step=global_step,
                stats=stats,
            )

        write_json(summary_path, stats)

    final_path = checkpoint_dir / "final.pt"
    if latest_path.exists():
        shutil.copy2(latest_path, final_path)
    print(f"[+] Training complete. latest={latest_path} best={best_path} summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
