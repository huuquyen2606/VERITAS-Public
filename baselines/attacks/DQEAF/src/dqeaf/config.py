from dataclasses import dataclass


@dataclass
class DQEAFConfig:
    # Algorithm-2 symbols from paper
    D: int = 30000
    T: int = 80
    F: int = 200
    TEST_INTERVAL: int = 1000
    MAX_RATIO: float = 7.0

    gamma: float = 0.99
    learning_rate: float = 1e-3
    minibatch_size: int = 32  # K
    replay_start_size: int = 1000  # B
    memory_capacity: int = 500000  # N (capacity of memory M)
    target_update_interval: int = 100  # U

    seed: int = 1337
    device: str = "cpu"

    # Optimization stability 
    grad_clip_norm: float = 10.0
    priority_epsilon: float = 1e-6
