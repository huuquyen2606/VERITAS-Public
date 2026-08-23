import gymnasium as gym


class ActionHistoryWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._history = []

    def reset(self, **kwargs):
        self._history = []
        obs, info = self.env.reset(**kwargs)
        info["action_history"] = list(self._history)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._history.append(action)
        info["action_history"] = list(self._history)
        return obs, reward, terminated, truncated, info
