import random

from obfumal.actions.base import Action


class OverlayAppend(Action):
    def __init__(self, min_log2: int = 5, max_log2: int = 8):
        self.min_log2 = min_log2
        self.max_log2 = max_log2

    @property
    def name(self) -> str:
        return "OverlayAppend"

    def apply(self, bytez: bytes) -> bytes:
        length = 2 ** random.randint(self.min_log2, self.max_log2)
        upper = random.randrange(256)
        extension = bytes(random.randint(0, upper) for _ in range(length))
        return bytez + extension
