from obfumal.reward.base import RewardFunction, RewardConfig


class ShapedReward(RewardFunction):
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
        if is_evaded:
            return self.config.evasion_reward
        return float(original_score - score)
