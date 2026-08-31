import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from obfumal.agents.dqn.model import QNetwork
from obfumal.agents.dqn.replay import ReplayBuffer
from obfumal.agents.dqn.schedule import LinearSchedule


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
        lr: float = 1e-3,
        gamma: float = 0.95,
        buffer_size: int = 1000,
        batch_size: int = 32,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 1000,
        target_update_freq: int = 100,
        hidden_dims: Optional[list] = None,
        policy: str = "boltzmann",
        temperature: float = 1.0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.policy = policy.lower()
        self.temperature = temperature

        self.q_network = QNetwork(state_dim, action_dim, hidden_dims=hidden_dims).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dims=hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, eps=1e-2)
        self.criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(buffer_size)

        self.epsilon_schedule = LinearSchedule(epsilon_start, epsilon_end, epsilon_decay_steps)
        self.steps_done = 0
        self.learn_steps = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        if self.policy == "boltzmann":
            self.q_network.eval()
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_t).squeeze(0)
                # Compute probability distribution over Q-values using Softmax
                probs = torch.softmax(q_values / self.temperature, dim=-1)
                # Sample an action from this distribution
                action = torch.multinomial(probs, num_samples=1).item()
            if not evaluate:
                self.q_network.train()
        else:
            # Fallback to standard epsilon-greedy policy
            eps = self.epsilon_schedule.value(self.steps_done)
            if evaluate or random.random() > eps:
                self.q_network.eval()
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.q_network(state_t)
                    action = q_values.argmax().item()
                if not evaluate:
                    self.q_network.train()
            else:
                action = random.randrange(self.action_dim)

        if not evaluate:
            self.steps_done += 1
        return action

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        self.q_network.train()
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        q_values = self.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_value = next_q_values.max(1)[0]
            expected_q_value = rewards + self.gamma * max_next_q_value * (~dones)

        loss = self.criterion(q_value, expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        return loss.item()

    def save(self, path: str):
        torch.save(
            {
                "model_state_dict": self.q_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.q_network.load_state_dict(checkpoint["model_state_dict"])
            self.target_network.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
