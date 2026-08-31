import lief

from obfumal.actions.base import Action
from obfumal.utils.lief_compat import build_binary_bytes, has_signature


class RemoveSignature(Action):
    @property
    def name(self) -> str:
        return "RemoveSignature"

    def apply(self, bytez: bytes) -> bytes:
        try:
            binary = lief.PE.parse(list(bytez))
        except (lief.bad_format, lief.read_out_of_bound, Exception):
            return bytez

        if not binary:
            return bytez

        if has_signature(binary):
            for directory in binary.data_directories:
                if directory.type == lief.PE.DATA_DIRECTORY.CERTIFICATE_TABLE:
                    directory.rva = 0
                    directory.size = 0
                    break

            built = build_binary_bytes(binary)
            return built if built is not None else bytez

        return bytez
