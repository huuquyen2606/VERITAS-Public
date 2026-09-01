import random


class EpsilonGreedyPolicy:
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def select(self, q_values, action_space_n: int) -> int:
        if random.random() < self.epsilon:
            return random.randrange(action_space_n)
        return int(q_values.argmax())
