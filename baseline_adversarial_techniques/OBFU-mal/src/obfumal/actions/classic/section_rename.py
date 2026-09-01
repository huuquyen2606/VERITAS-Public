import random

import lief

from obfumal.actions.base import Action
from obfumal.utils.lief_compat import build_binary_bytes

COMMON_SECTION_NAMES = [
    ".text",
    ".data",
    ".rdata",
    ".bss",
    ".tls",
    ".rsrc",
    ".idata",
    ".pdata",
    ".reloc",
    ".edata",
    ".CRT",
    ".INIT",
]


class SectionRename(Action):
    @property
    def name(self) -> str:
        return "SectionRename"

    def apply(self, bytez: bytes) -> bytes:
        try:
            binary = lief.PE.parse(list(bytez))
        except (lief.bad_format, lief.read_out_of_bound, Exception):
            return bytez

        if not binary or len(binary.sections) == 0:
            return bytez

        section = random.choice(binary.sections)
        new_name = random.choice(COMMON_SECTION_NAMES)
        section.name = new_name[:7]

        built = build_binary_bytes(binary)
        return built if built is not None else bytez
