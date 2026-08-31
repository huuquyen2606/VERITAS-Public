import time

import lief

from obfumal.actions.base import Action
from obfumal.utils.lief_compat import build_binary_bytes


class ChangeTimestamp(Action):
    @property
    def name(self) -> str:
        return "ChangeTimestamp"

    def apply(self, bytez: bytes) -> bytes:
        try:
            binary = lief.PE.parse(list(bytez))
        except (lief.bad_format, lief.read_out_of_bound, Exception):
            return bytez

        if not binary:
            return bytez

        binary.header.time_date_stamps = int(time.time())

        built = build_binary_bytes(binary)
        return built if built is not None else bytez
