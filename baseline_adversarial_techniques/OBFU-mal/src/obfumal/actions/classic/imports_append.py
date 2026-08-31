import random

import lief

from obfumal.actions.base import Action
from obfumal.utils.lief_compat import build_binary_bytes

COMMON_IMPORTS = {
    "KERNEL32.dll": [
        "GetTickCount",
        "GetCurrentProcessId",
        "GetLocalTime",
        "GetSystemTime",
        "GlobalAlloc",
        "VirtualAlloc",
        "CreateFileA",
        "CreateFileW",
    ],
    "USER32.dll": [
        "MessageBoxA",
        "MessageBoxW",
        "GetDesktopWindow",
        "GetForegroundWindow",
    ],
    "ADVAPI32.dll": [
        "RegOpenKeyExA",
        "RegQueryValueExA",
        "RegSetValueExA",
    ],
    "SHELL32.dll": [
        "ShellExecuteA",
        "ShellExecuteW",
    ],
    "WS2_32.dll": [
        "WSAStartup",
        "socket",
        "connect",
    ],
}


class ImportsAppend(Action):
    @property
    def name(self) -> str:
        return "ImportsAppend"

    def apply(self, bytez: bytes) -> bytes:
        try:
            binary = lief.PE.parse(list(bytez))
        except (lief.bad_format, lief.read_out_of_bound, Exception):
            return bytez

        if not binary:
            return bytez

        lib_name = random.choice(list(COMMON_IMPORTS.keys()))
        func_name = random.choice(list(COMMON_IMPORTS[lib_name]))

        lib = None
        for imported in binary.imports:
            if imported.name and imported.name.lower() == lib_name.lower():
                lib = imported
                break

        if lib is None:
            if hasattr(binary, "add_library"):
                lib = binary.add_library(lib_name)
            else:
                lib = binary.add_import(lib_name)

        existing = {entry.name for entry in lib.entries}
        if func_name not in existing:
            lib.add_entry(func_name)

        built = build_binary_bytes(binary, build_imports=True)
        return built if built is not None else bytez
