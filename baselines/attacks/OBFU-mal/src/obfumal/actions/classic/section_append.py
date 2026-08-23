import random

import lief

from obfumal.actions.base import Action
from obfumal.utils.lief_compat import build_binary_bytes


class SectionAppend(Action):
    def __init__(self, min_log2: int = 5, max_log2: int = 8):
        self.min_log2 = min_log2
        self.max_log2 = max_log2

    @property
    def name(self) -> str:
        return "SectionAppend"

    def apply(self, bytez: bytes) -> bytes:
        try:
            binary = lief.PE.parse(list(bytez))
        except (lief.bad_format, lief.read_out_of_bound, Exception):
            return bytez

        if not binary or not binary.sections:
            return bytez

        section = random.choice(binary.sections)
        length = 2 ** random.randint(self.min_log2, self.max_log2)
        available = section.size - len(section.content)
        if available <= 0:
            return bytez
        if length > available:
            length = available

        upper = random.randrange(256)
        base_content = list(section.content)
        section.content = base_content + [random.randint(0, upper) for _ in range(length)]

        built = build_binary_bytes(binary)
        return built if built is not None else bytez
