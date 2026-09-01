from __future__ import annotations

import random
from typing import Any

import gymnasium as gym
import numpy as np

from .actions import apply_action
from .classifier import IndependentClassifier
from .features import extract_raw_binary_features


class MalwareEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        malware_samples: list[bytes],
        classifier: IndependentClassifier,
        max_turn: int = 80,
        seed: int = 1337,
    ) -> None:
        super().__init__()
        if not malware_samples:
            raise ValueError("malware_samples cannot be empty")

        self.samples = malware_samples
        self.classifier = classifier
        self.max_turn = max_turn
        self.rng = random.Random(seed)

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=-0.5,
            high=0.5,
            shape=(513,),
            dtype=np.float32,
        )

        self._base_sample: bytes | None = None
        self._current_sample: bytes | None = None
        self._turn = 0

    def _current_obs(self) -> np.ndarray:
        if self._current_sample is None:
            raise RuntimeError("Environment is not reset")
        return extract_raw_binary_features(self._current_sample)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng.seed(seed)
        sample = self.rng.choice(self.samples)
        self._base_sample = sample
        self._current_sample = sample
        self._turn = 0
        return self._current_obs(), {}

    def reset_with_sample(self, sample: bytes) -> np.ndarray:
        self._base_sample = sample
        self._current_sample = sample
        self._turn = 0
        return self._current_obs()

    def step(self, action: int):
        if self._current_sample is None:
            raise RuntimeError("Environment is not reset")

        self._turn += 1
        self._current_sample = apply_action(self._current_sample, int(action), self.rng)

        malicious, score = self.classifier.predict_malicious(self._current_sample)

        # Equation (3) in paper.
        if not malicious:
            exponent = -((self._turn - 1) / self.max_turn)
            reward = (20.0 ** exponent) * 100.0
        else:
            reward = 0.0

        terminated = (not malicious) or (self._turn >= self.max_turn)
        truncated = False

        info = {
            "turn": self._turn,
            "malicious": malicious,
            "classifier_score": score,
        }
        return self._current_obs(), reward, terminated, truncated, info
