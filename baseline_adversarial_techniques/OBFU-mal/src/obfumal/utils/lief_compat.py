import os
import tempfile
from typing import Optional

import lief


def ensure_legacy_exception_aliases() -> None:
    # EMBER/gym-malware code paths still reference old top-level LIEF exception names.
    if not hasattr(lief, "not_found"):
        class _LiefNotFound(Exception):
            pass

        lief.not_found = _LiefNotFound  # type: ignore[attr-defined]

    for legacy_name in ("bad_file", "bad_format", "pe_error", "parser_error", "read_out_of_bound"):
        if not hasattr(lief, legacy_name):
            setattr(lief, legacy_name, RuntimeError)

    # EMBER expects lief.PE.SECTION_CHARACTERISTICS from older LIEF APIs.
    if not hasattr(lief.PE, "SECTION_CHARACTERISTICS") and hasattr(lief.PE.Section, "CHARACTERISTICS"):
        lief.PE.SECTION_CHARACTERISTICS = lief.PE.Section.CHARACTERISTICS  # type: ignore[attr-defined]


ensure_legacy_exception_aliases()


def _make_builder(binary: "lief.PE.Binary", build_imports: bool = False):
    try:
        return lief.PE.Builder(binary)
    except TypeError:
        config = lief.PE.Builder.config_t()
        if hasattr(config, "imports"):
            config.imports = bool(build_imports)
        return lief.PE.Builder(binary, config)


def build_binary_bytes(binary: "lief.PE.Binary", build_imports: bool = False) -> Optional[bytes]:
    try:
        builder = _make_builder(binary, build_imports=build_imports)
        if build_imports and hasattr(builder, "build_imports"):
            builder.build_imports(True)
        if build_imports and hasattr(builder, "patch_imports"):
            builder.patch_imports(True)
        builder.build()

        if hasattr(builder, "get_build"):
            return bytes(builder.get_build())
        if hasattr(builder, "bytes"):
            built = builder.bytes() if callable(builder.bytes) else builder.bytes
            return bytes(built)
        if hasattr(builder, "write"):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp_path = tmp.name
            try:
                builder.write(tmp_path)
                with open(tmp_path, "rb") as f:
                    return f.read()
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    except Exception:
        return None
    return None


def has_signature(binary: "lief.PE.Binary") -> bool:
    if hasattr(binary, "has_signature"):
        return bool(binary.has_signature)
    if hasattr(binary, "has_signatures"):
        return bool(binary.has_signatures)
    return False
