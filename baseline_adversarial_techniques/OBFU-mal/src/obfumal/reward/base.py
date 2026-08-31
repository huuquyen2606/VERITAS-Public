from abc import ABC, abstractmethod


class RewardFunction(ABC):
    @abstractmethod
    def __call__(
        self,
        *,
        original_score: float,
        score: float,
        is_evaded: bool,
        step: int,
        max_steps: int,
    ) -> float:
        raise NotImplementedError


class RewardConfig:
    def __init__(self, evasion_reward: float = 10.0, step_penalty: float = 0.0):
        self.evasion_reward = evasion_reward
        self.step_penalty = step_penalty
