import random
from typing import List, Sequence, Tuple


def train_test_split(sha256_list: Sequence[str], test_size: float = 0.2, seed: int = 123) -> Tuple[List[str], List[str]]:
    sha_list = list(sha256_list)
    rng = random.Random(seed)
    rng.shuffle(sha_list)
    split = int(len(sha_list) * (1.0 - test_size))
    return sha_list[:split], sha_list[split:]
