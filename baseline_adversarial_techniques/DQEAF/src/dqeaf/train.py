from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .classifier import ClassifierLike, save_classifier_artifact
from .config import DQEAFConfig
from .env import MalwareEnv
from .network import DQEAFNet
from .replay import PrioritizedReplay, Transition


class Trainer:
    def __init__(
        self,
        config: DQEAFConfig,
        classifier: ClassifierLike,
        train_malware_samples: list[bytes],
        test_malware_samples: list[bytes],
    ) -> None:
        self.cfg = config
        self.classifier = classifier
        self.train_samples = train_malware_samples
        self.test_samples = test_malware_samples

        if not self.train_samples:
            raise ValueError("train_malware_samples is empty")
        if not self.test_samples:
            raise ValueError("test_malware_samples is empty")

        self._seed_everything(self.cfg.seed)
        self.rng = random.Random(self.cfg.seed)

        self.env = MalwareEnv(
            malware_samples=self.train_samples,
            classifier=self.classifier,
            max_turn=self.cfg.T,
            seed=self.cfg.seed,
        )
        self.eval_env = MalwareEnv(
            malware_samples=self.test_samples,
            classifier=self.classifier,
            max_turn=self.cfg.T,
            seed=self.cfg.seed + 1,
        )

        self.device = torch.device(self.cfg.device)
        self.q_value = DQEAFNet(input_dim=513, action_dim=4).to(self.device)
        self.q_target = DQEAFNet(input_dim=513, action_dim=4).to(self.device)
        self.q_target.load_state_dict(self.q_value.state_dict())

        self.optimizer = torch.optim.Adam(self.q_value.parameters(), lr=self.cfg.learning_rate)
        self.memory = PrioritizedReplay(capacity=self.cfg.memory_capacity, seed=self.cfg.seed)
        self.total_train_steps = max(1, self.cfg.D * self.cfg.T)

        self.global_step = 0
        self.best_sr = -1.0
        self.history: list[dict[str, float]] = []

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _epsilon(self, n: int) -> float:
        # Paper Section IV-B.4: "epsilon annealed linearly from 1 to 0.1
        # over the first 1000 steps, and fixed at 0.1 thereafter."
        if n >= 1000:
            return 0.1
        return 1.0 - 0.9 * (n / 1000.0)

    def _select_action(self, state: np.ndarray, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return self.rng.randrange(4)
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q = self.q_value(s)
            return int(torch.argmax(q, dim=1).item())

    def _optimize_once(self) -> dict[str, float] | None:
        if len(self.memory) <= self.cfg.replay_start_size:
            return None

        batch, idx = self.memory.sample(self.cfg.minibatch_size)

        states = torch.from_numpy(np.stack([b.state for b in batch])).float().to(self.device)
        actions = torch.tensor([b.action for b in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(np.stack([b.next_state for b in batch])).float().to(self.device)
        dones = torch.tensor([b.done for b in batch], dtype=torch.float32, device=self.device)

        q_value_sa = self.q_value(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Paper Eq. (8): use target network for future reward term.
            next_actions = torch.argmax(self.q_value(next_states), dim=1)
            next_q_target = self.q_target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + self.cfg.gamma * next_q_target * (1.0 - dones)

        td_error = target - q_value_sa
        loss = F.mse_loss(q_value_sa, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_value.parameters(), self.cfg.grad_clip_norm)
        self.optimizer.step()

        new_p = td_error.detach().abs().cpu().numpy() + self.cfg.priority_epsilon
        self.memory.update_priorities(idx, new_p)

        return {
            "loss": float(loss.item()),
            "avg_td_error": float(np.mean(np.abs(new_p))),
        }

    def _run_testing_algorithm(self) -> dict[str, float]:
        # Algorithm 3 in paper.
        total_reward = 0.0
        success = 0
        f = min(self.cfg.F, len(self.test_samples))

        self.q_target.eval()
        with torch.no_grad():
            for i in range(f):
                state = self.eval_env.reset_with_sample(self.test_samples[i])
                episode_success = False
                for _ in range(self.cfg.T):
                    s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                    action = int(torch.argmax(self.q_target(s), dim=1).item())
                    state, reward, terminated, truncated, _info = self.eval_env.step(action)
                    total_reward += reward
                    if reward > 0:
                        episode_success = True
                    if terminated or truncated:
                        break
                if episode_success:
                    success += 1

        self.q_target.train()

        # Keep SR scale aligned with paper (0~10, threshold 7 == 70%).
        sr = (success / float(f)) * 10.0
        avg_reward = total_reward / float(f)
        return {"SR": sr, "avg_total_reward": avg_reward}

    def train(self) -> list[dict[str, float]]:
        for episode in range(1, self.cfg.D + 1):
            state, _ = self.env.reset()
            episode_reward = 0.0
            last_loss = 0.0
            print(f"[>] Bắt đầu Episode {episode}/{self.cfg.D}...")

            for _t in range(1, self.cfg.T + 1):
                epsilon = self._epsilon(self.global_step)
                action = self._select_action(state, epsilon)

                next_state, reward, terminated, truncated, _info = self.env.step(action)
                done = bool(terminated or truncated)

                transition = Transition(state=state, action=action, reward=reward, next_state=next_state, done=done)
                self.memory.add(transition, priority=self.memory.max_priority())

                opt = self._optimize_once()
                if opt is not None:
                    last_loss = opt["loss"]

                if self.global_step % self.cfg.target_update_interval == 0:
                    self.q_target.load_state_dict(self.q_value.state_dict())

                state = next_state
                episode_reward += reward
                self.global_step += 1

                if done:
                    break

            # Print per-episode logic always
            print(f"[-] Episode {episode}/{self.cfg.D} | Step {self.global_step} | Epsilon: {epsilon:.4f} | Reward: {episode_reward:.2f} | Loss: {last_loss:.4f}")

            log_item = {
                "episode": float(episode),
                "global_step": float(self.global_step),
                "epsilon": float(self._epsilon(self.global_step)),
                "episode_reward": float(episode_reward),
                "loss": float(last_loss),
            }

            if episode % self.cfg.TEST_INTERVAL == 0:
                test_stats = self._run_testing_algorithm()
                log_item.update(test_stats)

                if test_stats["SR"] > self.best_sr:
                    self.best_sr = test_stats["SR"]

                # Early stop rule from paper.
                if test_stats["SR"] > self.cfg.MAX_RATIO:
                    self.history.append(log_item)
                    break

            self.history.append(log_item)

        return self.history

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        torch.save(self.q_value.state_dict(), out / "q_value.pt")
        torch.save(self.q_target.state_dict(), out / "q_target.pt")
        save_classifier_artifact(self.classifier, out)

        metadata = {
            "config": asdict(self.cfg),
            "best_sr": self.best_sr,
            "steps": self.global_step,
            "history_len": len(self.history),
        }
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (out / "history.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
