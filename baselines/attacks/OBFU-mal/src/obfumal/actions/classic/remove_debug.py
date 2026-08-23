import lief

from obfumal.actions.base import Action
from obfumal.utils.lief_compat import build_binary_bytes


class RemoveDebug(Action):
    @property
    def name(self) -> str:
        return "RemoveDebug"

    def apply(self, bytez: bytes) -> bytes:
        try:
            binary = lief.PE.parse(list(bytez))
        except (lief.bad_format, lief.read_out_of_bound, Exception):
            return bytez

        if not binary:
            return bytez

        if binary.has_debug:
            for directory in binary.data_directories:
                if directory.type == lief.PE.DATA_DIRECTORY.DEBUG:
                    directory.rva = 0
                    directory.size = 0
                    break

            built = build_binary_bytes(binary)
            return built if built is not None else bytez

        return bytez
