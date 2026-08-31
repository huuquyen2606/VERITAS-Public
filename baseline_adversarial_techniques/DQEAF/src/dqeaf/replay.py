from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class PrioritizedReplay:
    def __init__(self, capacity: int, seed: int = 1337) -> None:
        self.capacity = capacity
        self.data: list[Transition] = []
        self.priorities: list[float] = []
        self.position = 0
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.data)

    def max_priority(self) -> float:
        if not self.priorities:
            return 1.0
        return max(self.priorities)

    def add(self, transition: Transition, priority: float | None = None) -> None:
        p = float(priority if priority is not None else self.max_priority())

        if len(self.data) < self.capacity:
            self.data.append(transition)
            self.priorities.append(p)
            return

        self.data[self.position] = transition
        self.priorities[self.position] = p
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> tuple[list[Transition], np.ndarray]:
        if len(self.data) < batch_size:
            raise ValueError("Not enough samples in replay buffer")

        probs = np.array(self.priorities, dtype=np.float64)
        probs /= probs.sum()

        idx = np.random.choice(len(self.data), size=batch_size, replace=False, p=probs)
        batch = [self.data[i] for i in idx]
        return batch, idx

    def update_priorities(self, indices: np.ndarray, new_priorities: np.ndarray) -> None:
        for i, p in zip(indices.tolist(), new_priorities.tolist()):
            self.priorities[i] = float(p)
