from obfumal.reward.base import RewardFunction, RewardConfig


class SparseReward(RewardFunction):
    def __init__(self, evasion_reward: float = 10.0):
        self.config = RewardConfig(evasion_reward=evasion_reward)

    def __call__(
        self,
        *,
        original_score: float,
        score: float,
        is_evaded: bool,
        step: int,
        max_steps: int,
    ) -> float:
        return self.config.evasion_reward if is_evaded else 0.0
