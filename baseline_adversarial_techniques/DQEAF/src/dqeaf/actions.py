from __future__ import annotations

import os
import random
import string
import tempfile
from enum import IntEnum

import lief
lief.logging.disable()


class Action(IntEnum):
    ARBE = 0
    ARI = 1
    ARS = 2
    RS = 3


_SECTION_TYPES = [
    "BSS",
    "UNKNOWN",
    "IDATA",
    "RELOC",
    "RSRC",
    "TEXT",
    "TLS",
]


def _random_ascii(rng: random.Random, min_len: int, max_len: int) -> str:
    n = rng.randint(min_len, max_len)
    letters = string.ascii_letters
    return "".join(rng.choice(letters) for _ in range(n))


def _parse_with_lief(data: bytes):
    fd, path = tempfile.mkstemp(suffix=".exe")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        binary = lief.parse(path)
        return binary, path
    except Exception:
        os.remove(path)
        raise


def _build_bytes(binary) -> bytes:
    fd, out_path = tempfile.mkstemp(suffix=".exe")
    os.close(fd)
    try:
        binary.write(out_path)
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


def _append_random_bytes(data: bytes, rng: random.Random) -> bytes:
    n = rng.randint(32, 512)
    suffix = bytes(rng.randint(0, 255) for _ in range(n))
    return data + suffix


def _append_random_import(data: bytes, rng: random.Random) -> bytes:
    binary, tmp = _parse_with_lief(data)
    try:
        lib_name = _random_ascii(rng, 5, 10).lower() + ".dll"
        func_name = _random_ascii(rng, 6, 16)
        imp = binary.add_library(lib_name)
        imp.add_entry(func_name)
        return _build_bytes(binary)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _section_characteristics(section_type: str):
    c = lief.PE.Section.CHARACTERISTICS
    if section_type == "BSS":
        return c.CNT_UNINITIALIZED_DATA | c.MEM_READ | c.MEM_WRITE
    if section_type == "IDATA":
        return c.CNT_INITIALIZED_DATA | c.MEM_READ
    if section_type == "RELOC":
        return c.CNT_INITIALIZED_DATA | c.MEM_READ | c.MEM_DISCARDABLE
    if section_type == "RSRC":
        return c.CNT_INITIALIZED_DATA | c.MEM_READ
    if section_type == "TEXT":
        return c.CNT_CODE | c.MEM_EXECUTE | c.MEM_READ
    if section_type == "TLS":
        return c.CNT_INITIALIZED_DATA | c.MEM_READ | c.MEM_WRITE
    return c.CNT_INITIALIZED_DATA | c.MEM_READ


def _append_random_section(data: bytes, rng: random.Random) -> bytes:
    binary, tmp = _parse_with_lief(data)
    try:
        sec_type = rng.choice(_SECTION_TYPES)
        sec_name = "." + _random_ascii(rng, 4, 7)
        sec = lief.PE.Section(sec_name)
        sec.content = [rng.randint(0, 255) for _ in range(rng.randint(128, 1024))]
        sec.characteristics = _section_characteristics(sec_type)
        binary.add_section(sec)
        return _build_bytes(binary)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _remove_signature(data: bytes) -> bytes:
    binary, tmp = _parse_with_lief(data)
    try:
        # Handle API changes across LIEF versions.
        if hasattr(binary, "remove_all_signatures"):
            binary.remove_all_signatures()
        elif hasattr(binary, "has_signatures") and binary.has_signatures:
            try:
                binary.signatures.clear()
            except Exception:
                pass

        # Remove certificate table entry if present.
        if hasattr(binary, "data_directories") and len(binary.data_directories) > 4:
            cert_dir = binary.data_directories[4]
            cert_dir.rva = 0
            cert_dir.size = 0
        return _build_bytes(binary)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _worker_apply_action(data: bytes, action: int, seed: int) -> bytes:
    """Apply one action from A={ARBE, ARI, ARS, RS} exactly as paper defines."""
    rng = random.Random(seed)
    act = Action(action)
    try:
        if act == Action.ARBE:
            return _append_random_bytes(data, rng)
        if act == Action.ARI:
            return _append_random_import(data, rng)
        if act == Action.ARS:
            return _append_random_section(data, rng)
        return _remove_signature(data)
    except Exception:
        # Keep pipeline alive when a PE edge case appears.
        return data

import concurrent.futures
import multiprocessing as mp

_action_pool: concurrent.futures.ProcessPoolExecutor | None = None

def _get_action_pool():
    global _action_pool
    if _action_pool is None:
        _action_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=1,
            mp_context=mp.get_context("spawn")
        )
    return _action_pool

def apply_action(data: bytes, action: int, rng: random.Random) -> bytes:
    """Robust wrapper to survive LIEF C++ Segmentation Faults during mutation."""
    global _action_pool
    pool = _get_action_pool()
    seed = rng.randint(0, 2**32 - 1)
    
    # Fast path for ARBE (appending random bytes) because it doesn't use LIEF parser at all.
    if Action(action) == Action.ARBE:
        return _worker_apply_action(data, action, seed)

    try:
        future = pool.submit(_worker_worker_trampoline, data, action, seed)
        return future.result(timeout=5.0)
    except concurrent.futures.TimeoutError:
        print(f"\n[-] Action LIEF Timeout (Action {action}): Process Hung. Restarting worker...")
        for pid in pool._processes.keys():
            import os, signal
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        pool.shutdown(wait=False)
        _action_pool = None
        return data
    except concurrent.futures.process.BrokenProcessPool:
        print(f"\n[-] Action LIEF Segfault (Action {action}): C++ Crashed! Restarting worker...")
        _action_pool = None
        return data
    except Exception as e:
        print(f"\n[-] Action exception: {e}")
        return data

def _worker_worker_trampoline(data: bytes, action: int, seed: int) -> bytes:
    """Top-level function for Windows/spawn mp compat"""
    return _worker_apply_action(data, action, seed)
