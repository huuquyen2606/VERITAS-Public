"""
Prioritized Experience Replay Buffer for VERITAS.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


class SumTree:
    """
    Binary sum tree for O(log N) prioritized sampling.
    """

    __slots__ = ("capacity", "tree")

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)

    @property
    def total_priority(self) -> float:
        """Total sum of all leaf priorities (stored at root)."""
        return float(self.tree[1])

    @property
    def max_leaf(self) -> float:
        """Maximum priority among all leaves."""
        return float(self.tree[self.capacity: 2 * self.capacity].max())

    def add(self, priority: float, leaf_pos: int) -> None:
        """Write ``priority`` at leaf ``leaf_pos`` and propagate up.

        Args:
            priority:   Priority value (must be > 0). Negative values are
                        clamped to a small positive epsilon to keep the tree
                        well-defined for prefix-sum sampling.
            leaf_pos:   Position in [0, capacity) - the circular pointer.
        """
        if priority < 0.0:
            priority = 1e-10
        tree_idx = leaf_pos + self.capacity
        self._update(tree_idx, priority)

    def update(self, tree_idx: int, priority: float) -> None:
        """Update an existing leaf's priority and propagate the change."""
        if priority < 0.0:
            priority = 1e-10
        self._update(tree_idx, priority)

    def get_leaf(self, value: float) -> Tuple[int, float, int]:
        """
        Retrieves the leaf whose prefix sum covers the given value.

        Args:
            value: A random number in [0, total_priority).

        Returns:
            Tuple of (tree_idx, priority, data_index).
        """
        idx = 1
        while idx < self.capacity:          
            left = 2 * idx
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        leaf_pos = idx - self.capacity
        return idx, self.tree[idx], leaf_pos

    def _update(self, tree_idx: int, priority: float) -> None:
        """Set leaf value and propagate delta up to root."""
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        idx = tree_idx >> 1
        while idx >= 1:
            self.tree[idx] += delta
            idx >>= 1


OBS_DIM = 2717

class PrioritizedReplayBuffer:
    """
    CPU-optimized Prioritized Experience Replay buffer.

    Args:
        capacity: Maximum number of transitions to store.
        obs_dim: Dimensionality of the observation space.
        alpha: Priority exponent.
        beta: Initial Importance Sampling exponent.
        beta_increment: Amount to increase beta per sample() call.
        epsilon: Small constant added to priorities.
    """

    def __init__(
        self,
        capacity: int = 10_000,
        obs_dim: int = OBS_DIM,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-5,
    ) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        self.states      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions     = np.zeros(capacity, dtype=np.int64)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones       = np.zeros(capacity, dtype=np.float32)

        self.tree = SumTree(capacity)

        self.write_ptr: int = 0
        self.size: int = 0
        self.max_priority: float = 1.0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        """Store a transition and assign it the current max priority.

        New transitions get ``max_priority`` so they are guaranteed to be
        sampled at least once before their priority is updated.

        Uses in-place copy into pre-allocated arrays — **no new allocation**.
        """
        idx = self.write_ptr

        self.states[idx]      = state
        self.actions[idx]     = action
        self.rewards[idx]     = reward
        self.next_states[idx] = next_state
        self.dones[idx]       = done

        priority = self.max_priority ** self.alpha
        self.tree.add(priority, leaf_pos=idx)

        self.write_ptr = (idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


    def sample(
        self, batch_size: int,
    ) -> Tuple[
        torch.Tensor,    
        torch.Tensor,    
        torch.Tensor,    
        torch.Tensor,    
        torch.Tensor,    
        torch.Tensor,    
        List[int],       
    ]:
        """
        Samples a mini-batch using stratified priority-proportional sampling.

        Returns:
            Tuple of 6 PyTorch tensors (states, actions, rewards, next_states,
            dones, weights) and a list of tree indices.
        """
        assert self.size >= batch_size, (
            f"Buffer has {self.size} transitions, need {batch_size}"
        )

        tree_indices: List[int] = []
        data_indices = np.empty(batch_size, dtype=np.int64)
        priorities   = np.empty(batch_size, dtype=np.float64)

        total = self.tree.total_priority
        segment = total / batch_size

        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            value = np.random.uniform(lo, hi)

            tree_idx, priority, data_idx = self.tree.get_leaf(value)
            tree_indices.append(tree_idx)
            data_indices[i] = data_idx
            priorities[i] = priority

        probabilities = priorities / total
        probabilities = np.clip(probabilities, 1e-10, None)

        weights = (self.size * probabilities) ** (-self.beta)
        weights /= weights.max()


        batch_states      = torch.from_numpy(self.states[data_indices].copy())
        batch_actions     = torch.from_numpy(self.actions[data_indices].copy())
        batch_rewards     = torch.from_numpy(self.rewards[data_indices].copy())
        batch_next_states = torch.from_numpy(self.next_states[data_indices].copy())
        batch_dones       = torch.from_numpy(self.dones[data_indices].copy())
        batch_weights     = torch.from_numpy(weights.astype(np.float32))

        return (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_dones,
            batch_weights,
            tree_indices,
        )

    def update_priorities(
        self,
        tree_indices: List[int],
        td_errors: np.ndarray,
    ) -> None:
        """
        Updates priorities after a training step.

        Args:
            tree_indices: List of tree node indices returned from sample().
            td_errors: Numpy array of per-sample TD errors with shape (B,).
        """
        for tree_idx, td_error in zip(tree_indices, td_errors):
            priority = (abs(float(td_error)) + self.epsilon) ** self.alpha
            self.tree.update(tree_idx, priority)
            self.max_priority = max(self.max_priority, abs(float(td_error)) + self.epsilon)

    def __len__(self) -> int:
        return self.size

    def memory_usage_mb(self) -> float:
        """Approximate memory usage of pre-allocated numpy arrays in MB."""
        bytes_per_float32 = 4
        bytes_per_int64 = 8
        total = (
            2 * self.capacity * self.obs_dim * bytes_per_float32
            + self.capacity * bytes_per_int64
            + 2 * self.capacity * bytes_per_float32
            + 2 * self.capacity * 8
        )
        return total / (1024 ** 2)


