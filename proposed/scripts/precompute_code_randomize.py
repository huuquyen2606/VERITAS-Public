"""Offline precomputation tool for CODE_RANDOMIZE (#15).

This script performs the heavy analysis phase outside the RL hot path. The
online action only loads the generated ``*_CR.exe`` file through
``context["precomputed_path"]``.

Implements all four in-place code-randomization transforms from
Pappas et al. 2012 ("Smashing the Gadgets", IEEE S&P §IV), adopted as the
``code_randomization`` macro-action by Song et al. 2021 (MAB-Malware §4.2.1):

  A.   Atomic instruction substitution (same-size, EFLAGS-aware whitelist).
  B.1. Intra basic-block reordering (RAW/WAR/WAW + memory/trap barriers).
  B.2. Register-preservation code reordering (prologue/epilogue push/pop).
  C.   Register reassignment (parallel self-contained live regions).

All patches are same-size and in-place: file size, section layout, and basic-
block boundaries stay byte-identical to the source PE. No new section, no
trampoline, no entry-point change. Pappas' approach is adapted to the
RetDec-free / IDA-Pro-free pipeline: angr ``CFGFast`` provides function and
basic-block boundaries; capstone ``regs_access()`` powers use/def; keystone
re-validates rewrites; LIEF maps RVA to file offset and reports relocations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import angr
except Exception as exc:  # pragma: no cover - dependency fallback
    angr = None
    _ANGR_IMPORT_ERROR = exc
else:
    _ANGR_IMPORT_ERROR = None

try:
    import lief
except Exception as exc:  # pragma: no cover - dependency fallback
    lief = None
    _LIEF_IMPORT_ERROR = exc
else:
    _LIEF_IMPORT_ERROR = None

try:
    from capstone import (
        CS_ARCH_X86,
        CS_GRP_CALL,
        CS_GRP_INT,
        CS_GRP_IRET,
        CS_GRP_JUMP,
        CS_GRP_RET,
        CS_MODE_32,
        CS_MODE_64,
        Cs,
        x86_const,
    )
    from capstone.x86 import X86_OP_IMM, X86_OP_MEM, X86_OP_REG, X86_REG_EIP, X86_REG_RIP
except Exception as exc:  # pragma: no cover - dependency fallback
    CS_ARCH_X86 = -1
    CS_MODE_32 = -1
    CS_MODE_64 = -1
    Cs = None
    x86_const = None
    X86_OP_IMM = None
    X86_OP_MEM = None
    X86_OP_REG = None
    X86_REG_EIP = -1
    X86_REG_RIP = -2
    CS_GRP_CALL = -1
    CS_GRP_INT = -2
    CS_GRP_IRET = -3
    CS_GRP_JUMP = -4
    CS_GRP_RET = -5
    _CAPSTONE_IMPORT_ERROR = exc
else:
    _CAPSTONE_IMPORT_ERROR = None

try:
    from keystone import KS_ARCH_X86, KS_MODE_32, KS_MODE_64, Ks
except Exception as exc:  # pragma: no cover - dependency fallback
    Ks = None
    _KEYSTONE_IMPORT_ERROR = exc
else:
    _KEYSTONE_IMPORT_ERROR = None


logger = logging.getLogger("precompute_code_randomize")

MACHINE_I386 = 0x014C
MACHINE_AMD64 = 0x8664
SUPPORTED_SUFFIXES = {".exe", ".dll", ".sys", ".ocx", ".scr"}
CONTROL_FLOW_GROUPS = {CS_GRP_CALL, CS_GRP_INT, CS_GRP_IRET, CS_GRP_JUMP, CS_GRP_RET}
ALU_DIRECTION_DUAL_OPCODES = {
    0x00: 0x02, 0x02: 0x00, 0x01: 0x03, 0x03: 0x01,  
    0x08: 0x0A, 0x0A: 0x08, 0x09: 0x0B, 0x0B: 0x09,  
    0x10: 0x12, 0x12: 0x10, 0x11: 0x13, 0x13: 0x11,  
    0x18: 0x1A, 0x1A: 0x18, 0x19: 0x1B, 0x1B: 0x19,  
    0x20: 0x22, 0x22: 0x20, 0x21: 0x23, 0x23: 0x21,  
    0x28: 0x2A, 0x2A: 0x28, 0x29: 0x2B, 0x2B: 0x29,  
    0x30: 0x32, 0x32: 0x30, 0x31: 0x33, 0x33: 0x31,  
    0x38: 0x3A, 0x3A: 0x38, 0x39: 0x3B, 0x3B: 0x39,  
}
LOGICAL_SAME_REG_MNEMONICS = ("test", "or", "and")
REORDER_ATTEMPTS = 16
_NOP_PADDING_TILES = (
    b"\x66\x0f\x1f\x84\x00\x00\x00\x00\x00",
    b"\x0f\x1f\x84\x00\x00\x00\x00\x00",
    b"\x0f\x1f\x80\x00\x00\x00\x00",
    b"\x66\x0f\x1f\x44\x00\x00",
    b"\x0f\x1f\x44\x00\x00",
    b"\x0f\x1f\x40\x00",
    b"\x0f\x1f\x00",
    b"\x66\x90",
    b"\x90",
)
TRAP_BARRIER_MNEMONICS = {
    "bound", "div", "idiv", "into", "ud2",
    "hlt", "cli", "sti", "in", "out",
    "lgdt", "lidt", "lldt", "lmsw", "ltr",
    "sgdt", "sidt", "sldt", "smsw", "str",
    "clts", "invd", "invlpg", "invlpga", "wbinvd",
    "rdmsr", "wrmsr", "rdpmc", "sysenter", "sysexit",
    "syscall", "sysret", "swapgs",
    "vmcall", "vmlaunch", "vmresume", "vmxoff",
    "xgetbv", "xsetbv", "xrstor", "xrstor64",
}


def _build_eflags_specs() -> Dict[str, Dict[str, int]]:
    if x86_const is None:
        return {}

    specs: Dict[str, Dict[str, int]] = {}
    for flag in ("CF", "PF", "AF", "ZF", "SF", "OF", "DF", "IF"):
        specs[flag.lower()] = {
            "read": (
                int(getattr(x86_const, f"X86_EFLAGS_TEST_{flag}", 0))
                | int(getattr(x86_const, f"X86_EFLAGS_PRIOR_{flag}", 0))
            ),
            "write": (
                int(getattr(x86_const, f"X86_EFLAGS_MODIFY_{flag}", 0))
                | int(getattr(x86_const, f"X86_EFLAGS_SET_{flag}", 0))
                | int(getattr(x86_const, f"X86_EFLAGS_RESET_{flag}", 0))
                | int(getattr(x86_const, f"X86_EFLAGS_UNDEFINED_{flag}", 0))
            ),
        }
    return specs


EFLAGS_SPECS = _build_eflags_specs()


@dataclass(frozen=True)
class CandidatePatch:
    """One same-size raw byte patch selected by an offline transform."""

    rva: int
    file_offset: int
    old_bytes: bytes
    new_bytes: bytes
    old_text: str
    new_text: str
    transform: str
    metadata: Optional[Dict[str, Any]] = None

    def to_manifest(self) -> Dict[str, Any]:
        payload = {
            "rva": self.rva,
            "file_offset": self.file_offset,
            "old_bytes": self.old_bytes.hex(),
            "new_bytes": self.new_bytes.hex(),
            "old_text": self.old_text,
            "new_text": self.new_text,
            "transform": self.transform,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass(frozen=True)
class InstructionDependency:
    """Minimal use/def model used by the intra-block scheduler."""

    index: int
    uses: Set[str]
    defs: Set[str]
    memory_access: bool
    barrier: bool


class AtomicInstructionSubstituter:
    """Conservative same-size instruction substitution pass."""

    def __init__(self, bits: int) -> None:
        mode = KS_MODE_64 if bits == 64 else KS_MODE_32
        cs_mode = CS_MODE_64 if bits == 64 else CS_MODE_32
        self.ks = Ks(KS_ARCH_X86, mode)
        self.md = Cs(CS_ARCH_X86, cs_mode)
        self.md.detail = True

    def collect(
        self,
        *,
        insn: Any,
        file_offset: int,
        raw_bytes: bytes,
        relocation_rvas: set[int],
        flags_dead_after: bool = False,
    ) -> Optional[CandidatePatch]:
        old_bytes = bytes(insn.bytes)
        size = int(insn.size)
        rva = int(insn.address)

        if size <= 0 or raw_bytes[file_offset:file_offset + size] != old_bytes:
            return None
        if old_bytes.startswith(b"\x66"):
            return None
        if any((rva + offset) in relocation_rvas for offset in range(size)):
            return None
        if _is_control_flow_instruction(insn):
            return None
        if len(getattr(insn, "operands", []) or []) != 2:
            return None

        same_reg_candidate = self._same_register_logical_candidate(
            insn=insn,
            file_offset=file_offset,
            old_bytes=old_bytes,
            size=size,
            rva=rva,
        )
        if same_reg_candidate is not None:
            return same_reg_candidate

        no_flags_candidate = self._no_flags_register_identity_candidate(
            insn=insn,
            file_offset=file_offset,
            old_bytes=old_bytes,
            size=size,
            rva=rva,
        )
        if no_flags_candidate is not None:
            return no_flags_candidate

        logical_identity_candidate = self._logical_identity_immediate_candidate(
            insn=insn,
            file_offset=file_offset,
            old_bytes=old_bytes,
            size=size,
            rva=rva,
        )
        if logical_identity_candidate is not None:
            return logical_identity_candidate

        inc_dec_candidate = self._dead_flags_inc_dec_candidate(
            insn=insn,
            file_offset=file_offset,
            old_bytes=old_bytes,
            size=size,
            rva=rva,
            flags_dead_after=flags_dead_after,
        )
        if inc_dec_candidate is not None:
            return inc_dec_candidate

        dead_flags_candidate = self._dead_flags_zeroing_candidate(
            insn=insn,
            file_offset=file_offset,
            old_bytes=old_bytes,
            size=size,
            rva=rva,
            flags_dead_after=flags_dead_after,
        )
        if dead_flags_candidate is not None:
            return dead_flags_candidate

        return self._direction_dual_candidate(
            insn=insn,
            file_offset=file_offset,
            old_bytes=old_bytes,
            size=size,
            rva=rva,
        )

    def _same_register_logical_candidate(
        self,
        *,
        insn: Any,
        file_offset: int,
        old_bytes: bytes,
        size: int,
        rva: int,
    ) -> Optional[CandidatePatch]:
        if len(getattr(insn, "operands", []) or []) != 2:
            return None
        op0, op1 = insn.operands
        if op0.type != X86_OP_REG or op1.type != X86_OP_REG or op0.reg != op1.reg:
            return None
        if insn.mnemonic not in LOGICAL_SAME_REG_MNEMONICS:
            return None

        reg_name = insn.reg_name(op0.reg)
        for replacement_mnemonic in LOGICAL_SAME_REG_MNEMONICS:
            if replacement_mnemonic == insn.mnemonic:
                continue
            new_text = f"{replacement_mnemonic} {reg_name}, {reg_name}"
            new_bytes = self._assemble_same_size(new_text, rva, size)
            new_insn = self._valid_single_non_branch(new_bytes, rva) if new_bytes else None
            if new_insn is None or new_bytes == old_bytes:
                continue
            if not _same_eflags_access(insn, new_insn):
                continue

            return CandidatePatch(
                rva=rva,
                file_offset=file_offset,
                old_bytes=old_bytes,
                new_bytes=new_bytes,
                old_text=_instruction_text(insn),
                new_text=new_text,
                transform="atomic_instruction_substitution",
                metadata={"rule": "same_register_logical"},
            )

        return None

    def _no_flags_register_identity_candidate(
        self,
        *,
        insn: Any,
        file_offset: int,
        old_bytes: bytes,
        size: int,
        rva: int,
    ) -> Optional[CandidatePatch]:
        if len(getattr(insn, "operands", []) or []) != 2:
            return None
        op0, op1 = insn.operands
        if insn.mnemonic != "lea" or op0.type != X86_OP_REG or op1.type != X86_OP_MEM:
            return None
        if op1.mem.base != op0.reg or op1.mem.index or int(op1.mem.disp) != 0:
            return None

        reg_name = insn.reg_name(op0.reg)
        new_text = f"mov {reg_name}, {reg_name}"
        new_bytes = self._assemble_padded_sequence(new_text, rva, size)
        new_insns = self._valid_non_branch_sequence(new_bytes, rva) if new_bytes else None
        if new_insns is None or new_bytes == old_bytes:
            return None
        if _eflags_access(new_insns[0]) != (set(), set()):
            return None

        return CandidatePatch(
            rva=rva,
            file_offset=file_offset,
            old_bytes=old_bytes,
            new_bytes=new_bytes,
            old_text=_instruction_text(insn),
            new_text=_sequence_text(new_insns),
            transform="atomic_instruction_substitution",
            metadata={"rule": "lea_identity_to_mov"},
        )

    def _logical_identity_immediate_candidate(
        self,
        *,
        insn: Any,
        file_offset: int,
        old_bytes: bytes,
        size: int,
        rva: int,
    ) -> Optional[CandidatePatch]:
        if len(getattr(insn, "operands", []) or []) != 2:
            return None
        op0, op1 = insn.operands
        if op0.type != X86_OP_REG or op1.type != X86_OP_IMM:
            return None
        if insn.mnemonic == "or" and int(op1.imm) != 0:
            return None
        if insn.mnemonic == "and" and int(op1.imm) not in {-1, 0xFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF}:
            return None
        if insn.mnemonic not in {"or", "and"}:
            return None

        reg_name = insn.reg_name(op0.reg)
        new_text = f"test {reg_name}, {reg_name}"
        new_bytes = self._assemble_padded_sequence(new_text, rva, size)
        new_insns = self._valid_non_branch_sequence(new_bytes, rva) if new_bytes else None
        if new_insns is None or new_bytes == old_bytes:
            return None
        if not _same_eflags_access(insn, new_insns[0]):
            return None

        return CandidatePatch(
            rva=rva,
            file_offset=file_offset,
            old_bytes=old_bytes,
            new_bytes=new_bytes,
            old_text=_instruction_text(insn),
            new_text=_sequence_text(new_insns),
            transform="atomic_instruction_substitution",
            metadata={"rule": "logical_identity_immediate"},
        )

    def _dead_flags_inc_dec_candidate(
        self,
        *,
        insn: Any,
        file_offset: int,
        old_bytes: bytes,
        size: int,
        rva: int,
        flags_dead_after: bool,
    ) -> Optional[CandidatePatch]:
        if not flags_dead_after:
            return None
        if len(getattr(insn, "operands", []) or []) != 2:
            return None
        op0, op1 = insn.operands
        if op0.type != X86_OP_REG or op1.type != X86_OP_IMM or int(op1.imm) != 1:
            return None
        replacement_mnemonic = {"add": "inc", "sub": "dec"}.get(insn.mnemonic)
        if replacement_mnemonic is None:
            return None

        reg_name = insn.reg_name(op0.reg)
        new_text = f"{replacement_mnemonic} {reg_name}"
        new_bytes = self._assemble_padded_sequence(new_text, rva, size)
        new_insns = self._valid_non_branch_sequence(new_bytes, rva) if new_bytes else None
        if new_insns is None or new_bytes == old_bytes:
            return None

        return CandidatePatch(
            rva=rva,
            file_offset=file_offset,
            old_bytes=old_bytes,
            new_bytes=new_bytes,
            old_text=_instruction_text(insn),
            new_text=_sequence_text(new_insns),
            transform="atomic_instruction_substitution",
            metadata={"rule": "dead_flags_inc_dec"},
        )

    def _dead_flags_zeroing_candidate(
        self,
        *,
        insn: Any,
        file_offset: int,
        old_bytes: bytes,
        size: int,
        rva: int,
        flags_dead_after: bool,
    ) -> Optional[CandidatePatch]:
        if not flags_dead_after:
            return None
        if len(getattr(insn, "operands", []) or []) != 2:
            return None
        op0, op1 = insn.operands
        if op0.type != X86_OP_REG or op1.type != X86_OP_REG or op0.reg != op1.reg:
            return None
        if insn.mnemonic not in {"xor", "sub"}:
            return None

        replacement_mnemonic = "sub" if insn.mnemonic == "xor" else "xor"
        reg_name = insn.reg_name(op0.reg)
        new_text = f"{replacement_mnemonic} {reg_name}, {reg_name}"
        new_bytes = self._assemble_same_size(new_text, rva, size)
        new_insn = self._valid_single_non_branch(new_bytes, rva) if new_bytes else None
        if new_insn is None or new_bytes == old_bytes:
            return None

        return CandidatePatch(
            rva=rva,
            file_offset=file_offset,
            old_bytes=old_bytes,
            new_bytes=new_bytes,
            old_text=_instruction_text(insn),
            new_text=new_text,
            transform="atomic_instruction_substitution",
            metadata={"rule": "dead_flags_zeroing"},
        )

    def _direction_dual_candidate(
        self,
        *,
        insn: Any,
        file_offset: int,
        old_bytes: bytes,
        size: int,
        rva: int,
    ) -> Optional[CandidatePatch]:
        if size != 2 or len(old_bytes) != 2:
            return None
        if 0x40 <= old_bytes[0] <= 0x4F:
            return None
        if len(getattr(insn, "operands", []) or []) != 2:
            return None
        op0, op1 = insn.operands
        if op0.type != X86_OP_REG or op1.type != X86_OP_REG:
            return None

        replacement_opcode = ALU_DIRECTION_DUAL_OPCODES.get(old_bytes[0])
        if replacement_opcode is None:
            return None

        modrm = old_bytes[1]
        if (modrm & 0xC0) != 0xC0:
            return None

        reg = (modrm >> 3) & 0x07
        rm = modrm & 0x07
        swapped_modrm = 0xC0 | (rm << 3) | reg
        new_bytes = bytes((replacement_opcode, swapped_modrm))
        new_insn = self._valid_single_non_branch(new_bytes, rva)
        if new_insn is None or new_bytes == old_bytes:
            return None
        if _normalize_instruction_text(new_insn) != _normalize_instruction_text(insn):
            return None
        if not _same_eflags_access(insn, new_insn):
            return None

        new_text = _instruction_text(new_insn)
        return CandidatePatch(
            rva=rva,
            file_offset=file_offset,
            old_bytes=old_bytes,
            new_bytes=new_bytes,
            old_text=_instruction_text(insn),
            new_text=new_text,
            transform="atomic_instruction_substitution",
            metadata={"rule": "direction_dual_opcode"},
        )

    def _assemble_same_size(self, asm_text: str, address: int, expected_size: int) -> Optional[bytes]:
        try:
            encoded, _ = self.ks.asm(asm_text, addr=address)
        except Exception as exc:
            logger.debug("Keystone rejected %r at %#x: %s", asm_text, address, exc)
            return None

        new_bytes = bytes(encoded)
        if len(new_bytes) != expected_size:
            return None
        return new_bytes

    def _assemble_padded_sequence(self, asm_text: str, address: int, expected_size: int) -> Optional[bytes]:
        try:
            encoded, _ = self.ks.asm(asm_text, addr=address)
        except Exception as exc:
            logger.debug("Keystone rejected %r at %#x: %s", asm_text, address, exc)
            return None

        core = bytes(encoded)
        if len(core) > expected_size:
            return None
        return core + _make_nop_padding(expected_size - len(core))

    def _valid_single_non_branch(self, code: bytes, address: int) -> Optional[Any]:
        decoded = list(self.md.disasm(code, address))
        if len(decoded) != 1:
            return None
        insn = decoded[0]
        if int(insn.size) != len(code) or _is_control_flow_instruction(insn):
            return None
        return insn

    def _valid_non_branch_sequence(self, code: bytes, address: int) -> Optional[List[Any]]:
        decoded = list(self.md.disasm(code, address))
        if not decoded or sum(int(insn.size) for insn in decoded) != len(code):
            return None
        if _is_control_flow_instruction(decoded[0]):
            return None
        if any(not _is_nop_instruction(insn) for insn in decoded[1:]):
            return None
        return decoded


class IntraBlockReorderer:
    """Conservative intra-basic-block instruction reordering."""

    def __init__(self, bits: int) -> None:
        cs_mode = CS_MODE_64 if bits == 64 else CS_MODE_32
        self.md = Cs(CS_ARCH_X86, cs_mode)
        self.md.detail = True

    def collect(
        self,
        *,
        block: Any,
        binary: Any,
        raw_bytes: bytes,
        relocation_rvas: set[int],
        rng: random.Random,
    ) -> List[CandidatePatch]:
        insns = list(getattr(getattr(block, "capstone", None), "insns", []) or [])
        if len(insns) < 3:
            return []

        terminator = insns[-1] if _is_control_flow_instruction(insns[-1]) else None
        body = insns[:-1] if terminator is not None else insns
        if len(body) < 2:
            return []
        if any(_is_control_flow_instruction(insn) for insn in body):
            return []

        file_offsets: List[int] = []
        for insn in insns:
            size = int(insn.size)
            rva = int(insn.address)
            old_bytes = bytes(insn.bytes)
            if size <= 0 or old_bytes.startswith(b"\x66"):
                return []
            if any((rva + offset) in relocation_rvas for offset in range(size)):
                return []
            if _has_pc_relative_operand(insn) or _has_complex_implicit_memory(insn):
                return []

            file_offset = _rva_to_file_offset(binary, rva, size)
            if file_offset is None:
                return []
            if raw_bytes[file_offset:file_offset + size] != old_bytes:
                return []
            file_offsets.append(file_offset)

        for left, right in zip(range(len(insns) - 1), range(1, len(insns))):
            expected_next = file_offsets[left] + int(insns[left].size)
            if file_offsets[right] != expected_next:
                return []

        dependencies = [_instruction_dependency(index, insn) for index, insn in enumerate(body)]
        edges = _dependency_edges(dependencies)
        if all(info.barrier for info in dependencies):
            return []
        original_order = list(range(len(body)))
        new_order = _random_topological_order(len(body), edges, original_order, rng)
        if new_order is None:
            return []

        reordered_body = [body[index] for index in new_order]
        new_insns = reordered_body + ([terminator] if terminator is not None else [])
        old_start = file_offsets[0]
        old_end = file_offsets[-1] + int(insns[-1].size)
        old_block_bytes = raw_bytes[old_start:old_end]
        expected_old_bytes = b"".join(bytes(insn.bytes) for insn in insns)
        if old_block_bytes != expected_old_bytes:
            return []

        new_block_bytes = b"".join(bytes(insn.bytes) for insn in new_insns)
        if len(new_block_bytes) != len(old_block_bytes) or new_block_bytes == old_block_bytes:
            return []
        if not self._valid_reordered_block(new_block_bytes, int(insns[0].address), len(new_insns), terminator):
            return []

        dependency_edge_count = sum(len(targets) for targets in edges.values())
        return [
            CandidatePatch(
                rva=int(insns[0].address),
                file_offset=old_start,
                old_bytes=old_block_bytes,
                new_bytes=new_block_bytes,
                old_text=" | ".join(_instruction_text(insn) for insn in insns),
                new_text=" | ".join(_instruction_text(insn) for insn in new_insns),
                transform="intra_block_reordering",
                metadata={
                    "block_rva": int(insns[0].address),
                    "dependency_edges": dependency_edge_count,
                    "order_before": [_instruction_text(insn) for insn in body],
                    "order_after": [_instruction_text(body[index]) for index in new_order],
                },
            )
        ]

    def _valid_reordered_block(
        self,
        new_block_bytes: bytes,
        address: int,
        expected_count: int,
        terminator: Optional[Any],
    ) -> bool:
        decoded = list(self.md.disasm(new_block_bytes, address))
        if len(decoded) != expected_count:
            return False
        if sum(int(insn.size) for insn in decoded) != len(new_block_bytes):
            return False
        if terminator is not None:
            if _normalize_instruction_text(decoded[-1]) != _normalize_instruction_text(terminator):
                return False
        return True




_PUSH_OPCODE_BASE = 0x50
_POP_OPCODE_BASE = 0x58
_GP_REG_NAMES_32 = {0: "eax", 1: "ecx", 2: "edx", 3: "ebx", 4: "esp", 5: "ebp", 6: "esi", 7: "edi"}
_GP_REG_NAMES_64 = {0: "rax", 1: "rcx", 2: "rdx", 3: "rbx", 4: "rsp", 5: "rbp", 6: "rsi", 7: "rdi"}
_FRAME_REG_INDEXES = {4}


class RegisterPreservationReorderer:
    """Pappas IV.B.2: reorder ``push <callee-saved>`` in the prologue and
    mirror the change in every epilogue's matching ``pop`` run."""

    def __init__(self, bits: int) -> None:
        self.bits = bits
        cs_mode = CS_MODE_64 if bits == 64 else CS_MODE_32
        self.md = Cs(CS_ARCH_X86, cs_mode)
        self.md.detail = True

    def collect(
        self,
        *,
        func: Any,
        binary: Any,
        raw_bytes: bytes,
        relocation_rvas: set[int],
        rng: random.Random,
    ) -> List[CandidatePatch]:
        try:
            entry_block = func.get_block(func.addr)
        except Exception:
            return []
        if entry_block is None:
            return []

        prologue = self._scan_prologue(func, entry_block, raw_bytes, binary, relocation_rvas)
        if prologue is None:
            return []
        prologue_regs, prologue_offsets = prologue
        if len(prologue_regs) < 2:
            return []

        epilogue_offsets: List[List[int]] = []
        for end_node in self._iter_function_terminators(func):
            ep = self._scan_epilogue(end_node, raw_bytes, binary, relocation_rvas)
            if ep is None:
                return []
            ep_regs, ep_offsets = ep
            if list(reversed(prologue_regs)) != ep_regs:
                return []
            epilogue_offsets.append(ep_offsets)

        if not epilogue_offsets:
            return []

        original_order = list(range(len(prologue_regs)))
        new_order = list(original_order)
        for _ in range(REORDER_ATTEMPTS):
            rng.shuffle(new_order)
            if new_order != original_order:
                break
        if new_order == original_order:
            return []

        new_push_regs = [prologue_regs[i] for i in new_order]
        new_pop_regs = list(reversed(new_push_regs))

        patches: List[CandidatePatch] = []

        old_prologue_bytes = bytes(_PUSH_OPCODE_BASE + r for r in prologue_regs)
        new_prologue_bytes = bytes(_PUSH_OPCODE_BASE + r for r in new_push_regs)
        if old_prologue_bytes == new_prologue_bytes:
            return []

        prologue_start = prologue_offsets[0]
        if raw_bytes[prologue_start:prologue_start + len(old_prologue_bytes)] != old_prologue_bytes:
            return []
        patches.append(
            CandidatePatch(
                rva=int(entry_block.addr) if hasattr(entry_block, "addr") else int(func.addr),
                file_offset=prologue_start,
                old_bytes=old_prologue_bytes,
                new_bytes=new_prologue_bytes,
                old_text=" ; ".join(f"push {self._reg_name(r)}" for r in prologue_regs),
                new_text=" ; ".join(f"push {self._reg_name(r)}" for r in new_push_regs),
                transform="reg_preservation_reorder",
                metadata={
                    "function_addr": int(func.addr),
                    "prologue_old": [self._reg_name(r) for r in prologue_regs],
                    "prologue_new": [self._reg_name(r) for r in new_push_regs],
                    "epilogue_count": len(epilogue_offsets),
                },
            )
        )

        original_pop_regs = list(reversed(prologue_regs))
        old_pop_bytes = bytes(_POP_OPCODE_BASE + r for r in original_pop_regs)
        new_pop_bytes = bytes(_POP_OPCODE_BASE + r for r in new_pop_regs)
        for ep_offsets in epilogue_offsets:
            ep_start = ep_offsets[0]
            if raw_bytes[ep_start:ep_start + len(old_pop_bytes)] != old_pop_bytes:
                return []
            patches.append(
                CandidatePatch(
                    rva=ep_start,  
                    file_offset=ep_start,
                    old_bytes=old_pop_bytes,
                    new_bytes=new_pop_bytes,
                    old_text=" ; ".join(f"pop {self._reg_name(r)}" for r in original_pop_regs),
                    new_text=" ; ".join(f"pop {self._reg_name(r)}" for r in new_pop_regs),
                    transform="reg_preservation_reorder",
                    metadata={
                        "function_addr": int(func.addr),
                        "epilogue_old": [self._reg_name(r) for r in original_pop_regs],
                        "epilogue_new": [self._reg_name(r) for r in new_pop_regs],
                        "kind": "epilogue",
                    },
                )
            )
        return patches


    def _reg_name(self, reg_idx: int) -> str:
        table = _GP_REG_NAMES_64 if self.bits == 64 else _GP_REG_NAMES_32
        return table.get(reg_idx, f"r{reg_idx}")

    def _has_frame_setup(self, entry_block: Any) -> bool:
        """Detect ``push ebp/rbp ; mov ebp/rbp, esp/rsp`` in the first few
        instructions of a function. Such functions use EBP/RBP as a frame
        pointer; reordering the prologue would shift the EBP-relative
        local-variable layout."""
        try:
            insns = list(entry_block.capstone.insns)
        except Exception:
            return True  
        if not insns:
            return True
        head = insns[: min(8, len(insns))]
        seen_push_ebp = False
        for insn in head:
            if insn.mnemonic == "push" and len(bytes(insn.bytes)) == 1:
                opcode = insn.bytes[0]
                if opcode in (_PUSH_OPCODE_BASE + 5,):  
                    seen_push_ebp = True
            elif insn.mnemonic == "mov":
                op_str = insn.op_str.lower()
                if seen_push_ebp and any(
                    pair in op_str
                    for pair in ("ebp, esp", "rbp, rsp")
                ):
                    return True
            elif insn.mnemonic == "leave":
                return True
        return False

    def _iter_function_terminators(self, func: Any) -> Iterable[Any]:
        results: List[Any] = []
        endpoints = getattr(func, "endpoints", None)
        if endpoints:
            for node in endpoints:
                try:
                    block = func.get_block(int(node.addr))
                except Exception:
                    continue
                if block is not None:
                    results.append(block)
            if results:
                return results
        for block in func.blocks:
            try:
                cap_insns = list(block.capstone.insns)
            except Exception:
                continue
            if cap_insns and _is_control_flow_instruction(cap_insns[-1]) and cap_insns[-1].mnemonic in ("ret", "retn", "retf"):
                results.append(block)
        return results

    def _scan_prologue(
        self,
        func: Any,
        entry_block: Any,
        raw_bytes: bytes,
        binary: Any,
        relocation_rvas: set[int],
    ) -> Optional[Tuple[List[int], List[int]]]:
        try:
            insns = list(entry_block.capstone.insns)
        except Exception:
            return None
        cursor = 0
        if (
            len(insns) >= 2
            and insns[0].mnemonic == "push"
            and len(bytes(insns[0].bytes)) == 1
            and insns[0].bytes[0] == _PUSH_OPCODE_BASE + 5  
            and insns[1].mnemonic == "mov"
            and any(
                pair in insns[1].op_str.lower()
                for pair in ("ebp, esp", "rbp, rsp")
            )
        ):
            cursor = 2
        return self._collect_consecutive_pushpop(
            insns[cursor:],
            opcode_base=_PUSH_OPCODE_BASE,
            mnemonic="push",
            raw_bytes=raw_bytes,
            binary=binary,
            relocation_rvas=relocation_rvas,
        )

    def _scan_epilogue(
        self,
        block: Any,
        raw_bytes: bytes,
        binary: Any,
        relocation_rvas: set[int],
    ) -> Optional[Tuple[List[int], List[int]]]:
        try:
            insns = list(block.capstone.insns)
        except Exception:
            return None
        if not insns or insns[-1].mnemonic not in ("ret", "retn", "retf"):
            return None
        body = list(insns[:-1])
        if body and body[-1].mnemonic == "leave":
            body.pop()
        else:
            if (
                len(body) >= 2
                and body[-1].mnemonic == "pop"
                and len(bytes(body[-1].bytes)) == 1
                and body[-1].bytes[0] == _POP_OPCODE_BASE + 5  
                and body[-2].mnemonic == "mov"
                and any(
                    pair in body[-2].op_str.lower()
                    for pair in ("esp, ebp", "rsp, rbp")
                )
            ):
                body = body[:-2]
        run: List[Any] = []
        for insn in reversed(body):
            if insn.mnemonic != "pop":
                break
            run.append(insn)
        if not run:
            return None
        run.reverse()
        return self._collect_consecutive_pushpop(
            run,
            opcode_base=_POP_OPCODE_BASE,
            mnemonic="pop",
            raw_bytes=raw_bytes,
            binary=binary,
            relocation_rvas=relocation_rvas,
        )

    def _collect_consecutive_pushpop(
        self,
        insns: Sequence[Any],
        *,
        opcode_base: int,
        mnemonic: str,
        raw_bytes: bytes,
        binary: Any,
        relocation_rvas: set[int],
    ) -> Optional[Tuple[List[int], List[int]]]:
        regs: List[int] = []
        offsets: List[int] = []
        prev_offset: Optional[int] = None
        prev_size: Optional[int] = None
        for insn in insns:
            if insn.mnemonic != mnemonic:
                break
            insn_bytes = bytes(insn.bytes)
            if len(insn_bytes) != 1:
                return None  
            opcode = insn_bytes[0]
            reg_idx = opcode - opcode_base
            if not 0 <= reg_idx < 8 or reg_idx in _FRAME_REG_INDEXES:
                return None
            rva = int(insn.address)
            if rva in relocation_rvas:
                return None
            file_offset = _rva_to_file_offset(binary, rva, 1)
            if file_offset is None:
                return None
            if raw_bytes[file_offset:file_offset + 1] != insn_bytes:
                return None
            if prev_offset is not None and file_offset != prev_offset + prev_size:
                return None
            regs.append(reg_idx)
            offsets.append(file_offset)
            prev_offset = file_offset
            prev_size = 1
        if not regs:
            return None
        return regs, offsets




_REASSIGN_PAIRS_32 = (
    ("eax", "ebx"), ("eax", "ecx"), ("eax", "edx"),
    ("ebx", "ecx"), ("ebx", "edx"), ("ecx", "edx"),
)
_REASSIGN_PAIRS_64 = (
    ("rax", "rbx"), ("rax", "rcx"), ("rax", "rdx"),
    ("rbx", "rcx"), ("rbx", "rdx"), ("rcx", "rdx"),
)
_REG_ALIAS_GROUPS = {
    "rax": ("rax", "eax", "ax", "ah", "al"),
    "rbx": ("rbx", "ebx", "bx", "bh", "bl"),
    "rcx": ("rcx", "ecx", "cx", "ch", "cl"),
    "rdx": ("rdx", "edx", "dx", "dh", "dl"),
    "eax": ("eax", "ax", "ah", "al"),
    "ebx": ("ebx", "bx", "bh", "bl"),
    "ecx": ("ecx", "cx", "ch", "cl"),
    "edx": ("edx", "dx", "dh", "dl"),
}
_FRAME_REG_NAMES = {"esp", "rsp", "ebp", "rbp"}
_IMPLICIT_REG_MNEMONICS = {
    "movs", "movsb", "movsw", "movsd", "movsq",
    "stos", "stosb", "stosw", "stosd", "stosq",
    "lods", "lodsb", "lodsw", "lodsd", "lodsq",
    "scas", "scasb", "scasw", "scasd", "scasq",
    "cmps", "cmpsb", "cmpsw", "cmpsd", "cmpsq",
    "ins", "outs",
    "mul", "imul", "div", "idiv", "in", "out",
    "cpuid", "xlat", "xlatb",
    "enter", "leave",
}


class RegisterReassigner:
    """Pappas IV.C: swap two GPR names across a per-block self-contained
    live region."""

    def __init__(self, bits: int) -> None:
        self.bits = bits
        cs_mode = CS_MODE_64 if bits == 64 else CS_MODE_32
        ks_mode = KS_MODE_64 if bits == 64 else KS_MODE_32
        self.md = Cs(CS_ARCH_X86, cs_mode)
        self.md.detail = True
        self.ks = Ks(KS_ARCH_X86, ks_mode)
        self.pairs: Sequence[Tuple[str, str]] = (
            _REASSIGN_PAIRS_64 if bits == 64 else _REASSIGN_PAIRS_32
        )

    def collect(
        self,
        *,
        func: Any,
        binary: Any,
        raw_bytes: bytes,
        relocation_rvas: set[int],
        rng: random.Random,
    ) -> List[CandidatePatch]:
        patches: List[CandidatePatch] = []
        for block in self._iter_function_blocks(func):
            try:
                insns = list(block.capstone.insns)
            except Exception:
                continue
            if len(insns) < 2:
                continue
            if any(insn.mnemonic in _IMPLICIT_REG_MNEMONICS for insn in insns):
                continue
            if any(self._touches_frame(insn) for insn in insns):
                continue
            block_bytes_offsets: List[int] = []
            ok = True
            for insn in insns:
                rva = int(insn.address)
                size = int(insn.size)
                insn_bytes = bytes(insn.bytes)
                if size <= 0 or insn_bytes.startswith(b"\x66"):
                    ok = False
                    break
                if any((rva + offset) in relocation_rvas for offset in range(size)):
                    ok = False
                    break
                file_offset = _rva_to_file_offset(binary, rva, size)
                if file_offset is None:
                    ok = False
                    break
                if raw_bytes[file_offset:file_offset + size] != insn_bytes:
                    ok = False
                    break
                block_bytes_offsets.append(file_offset)
            if not ok or not block_bytes_offsets:
                continue
            for left, right in zip(range(len(insns) - 1), range(1, len(insns))):
                if block_bytes_offsets[right] != block_bytes_offsets[left] + int(insns[left].size):
                    ok = False
                    break
            if not ok:
                continue

            shuffled_pairs = list(self.pairs)
            rng.shuffle(shuffled_pairs)
            for r1, r2 in shuffled_pairs:
                if not self._pair_self_contained_in_block(insns, r1, r2):
                    continue
                rewritten = self._rewrite_block(insns, r1, r2)
                if rewritten is None:
                    continue
                new_block_bytes = rewritten
                old_start = block_bytes_offsets[0]
                old_end = block_bytes_offsets[-1] + int(insns[-1].size)
                old_block_bytes = bytes(raw_bytes[old_start:old_end])
                if len(new_block_bytes) != len(old_block_bytes):
                    continue
                if new_block_bytes == old_block_bytes:
                    continue
                if not self._validate_rewrite(new_block_bytes, int(insns[0].address), len(insns)):
                    continue
                patches.append(
                    CandidatePatch(
                        rva=int(insns[0].address),
                        file_offset=old_start,
                        old_bytes=old_block_bytes,
                        new_bytes=new_block_bytes,
                        old_text=_sequence_text(insns),
                        new_text=f"<{r1}<->{r2}> " + _sequence_text(insns),
                        transform="register_reassignment",
                        metadata={
                            "function_addr": int(func.addr),
                            "block_rva": int(insns[0].address),
                            "swap_pair": [r1, r2],
                            "instruction_count": len(insns),
                        },
                    )
                )
                break  
        return patches


    def _iter_function_blocks(self, func: Any) -> Iterable[Any]:
        try:
            return list(func.blocks)
        except Exception:
            return []

    def _touches_frame(self, insn: Any) -> bool:
        regs_read, regs_write = insn.regs_access()
        for reg_id in tuple(regs_read) + tuple(regs_write):
            try:
                name = insn.reg_name(reg_id)
            except Exception:
                continue
            if name and name.lower() in _FRAME_REG_NAMES:
                return True
        return False

    def _used_register_names(self, insn: Any) -> Set[str]:
        names: Set[str] = set()
        regs_read, regs_write = insn.regs_access()
        for reg_id in tuple(regs_read) + tuple(regs_write):
            try:
                name = insn.reg_name(reg_id)
            except Exception:
                continue
            if name:
                names.add(name.lower())
        return names

    def _pair_self_contained_in_block(self, insns: Sequence[Any], r1: str, r2: str) -> bool:
        r1_alias = set(_REG_ALIAS_GROUPS.get(r1, (r1,)))
        r2_alias = set(_REG_ALIAS_GROUPS.get(r2, (r2,)))
        r1_seen_write_first = None
        r2_seen_write_first = None
        for insn in insns:
            regs_read, regs_write = insn.regs_access()
            read_names = {insn.reg_name(rid) or "" for rid in regs_read}
            write_names = {insn.reg_name(rid) or "" for rid in regs_write}
            read_names = {n.lower() for n in read_names if n}
            write_names = {n.lower() for n in write_names if n}

            if r1_seen_write_first is None and (read_names & r1_alias or write_names & r1_alias):
                r1_seen_write_first = bool(write_names & r1_alias) and not bool(read_names & r1_alias)
            if r2_seen_write_first is None and (read_names & r2_alias or write_names & r2_alias):
                r2_seen_write_first = bool(write_names & r2_alias) and not bool(read_names & r2_alias)
        return bool(r1_seen_write_first) and bool(r2_seen_write_first)

    def _rewrite_block(self, insns: Sequence[Any], r1: str, r2: str) -> Optional[bytes]:
        new_insns: List[bytes] = []
        for insn in insns:
            text = f"{insn.mnemonic} {insn.op_str}".strip()
            new_text = self._swap_register_tokens(text, r1, r2)
            if new_text == text:
                new_insns.append(bytes(insn.bytes))
                continue
            try:
                encoded, _ = self.ks.asm(new_text, addr=int(insn.address))
            except Exception:
                return None
            if encoded is None:
                return None
            encoded_bytes = bytes(encoded)
            if len(encoded_bytes) != int(insn.size):
                return None
            new_insns.append(encoded_bytes)
        return b"".join(new_insns)

    def _swap_register_tokens(self, text: str, r1: str, r2: str) -> str:
        aliases1 = _REG_ALIAS_GROUPS.get(r1, (r1,))
        aliases2 = _REG_ALIAS_GROUPS.get(r2, (r2,))
        result = text
        for w1, w2 in zip(aliases1, aliases2):
            sentinel = f"\x00__SWAP_{w1}__\x00"
            result = re.sub(rf"\b{re.escape(w1)}\b", sentinel, result)
            result = re.sub(rf"\b{re.escape(w2)}\b", w1, result)
            result = result.replace(sentinel, w2)
        return result

    def _validate_rewrite(self, new_block_bytes: bytes, address: int, expected_count: int) -> bool:
        decoded = list(self.md.disasm(new_block_bytes, address))
        if len(decoded) != expected_count:
            return False
        if sum(int(insn.size) for insn in decoded) != len(new_block_bytes):
            return False
        return True


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute CODE_RANDOMIZE variants for PE files.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Single PE file to process.")
    source.add_argument("--input-dir", help="Directory of PE files to process.")
    parser.add_argument(
        "--output-dir",
        default="data/precomputed/code-randomize",
        help="Directory for *_CR.exe outputs.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Deterministic patch selection seed.")
    parser.add_argument("--max-rewrites", type=int, default=200, help="Maximum CODE_RANDOMIZE patches per sample.")
    parser.add_argument("--timeout-sec", type=int, default=180, help="Per-sample soft timeout.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    missing = _missing_dependencies()
    if missing:
        logger.error("Missing dependencies: %s", ", ".join(missing))
        return 1

    output_dir = Path(args.output_dir)
    (output_dir / "exe").mkdir(parents=True, exist_ok=True)
    (output_dir / "json").mkdir(parents=True, exist_ok=True)

    input_paths = list(_iter_inputs(args))
    if not input_paths:
        logger.error("No input PE files found.")
        return 1

    failures = 0
    for path in input_paths:
        report = process_file(path, output_dir, args)
        status = report["status"]
        if status not in {"changed", "skipped_no_candidates"}:
            failures += 1
        logger.info("%s: %s (%s patches)", path.name, status, report.get("patch_count", 0))

    return 1 if failures == len(input_paths) else 0


def process_file(path: Path, output_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    started = time.monotonic()
    deadline = started + max(1, int(args.timeout_sec))
    report: Dict[str, Any] = {
        "input_path": str(path),
        "output_path": None,
        "status": "failed",
        "reason": None,
        "patch_count": 0,
        "patches": [],
        "elapsed_sec": 0.0,
    }

    try:
        original = path.read_bytes()
        report["sha256_original"] = _sha256(original)

        bits = _pe_bits_from_bytes(original)
        report["bits"] = bits

        binary = lief.PE.parse(str(path))
        if binary is None:
            report.update(status="skipped_parse_failed", reason="LIEF returned None")
            return _write_report(report, output_dir, path, started)

        _check_deadline(deadline)
        relocation_rvas = _relocation_rvas(binary)
        rng = random.Random(_sample_seed(args.seed, original))
        candidates = _collect_candidates(
            path=path,
            binary=binary,
            bits=bits,
            raw_bytes=original,
            relocation_rvas=relocation_rvas,
            deadline=deadline,
            rng=rng,
        )

        if not candidates:
            report.update(status="skipped_no_candidates", reason="No safe CODE_RANDOMIZE candidates found")
            return _write_report(report, output_dir, path, started)

        selected = _select_non_overlapping(candidates, max(0, int(args.max_rewrites)), rng)
        if not selected:
            report.update(status="skipped_no_candidates", reason="No non-overlapping candidate survived selection")
            return _write_report(report, output_dir, path, started)

        mutated = _apply_patches(original, selected)
        if mutated == original:
            report.update(status="skipped_no_change", reason="Selected patches did not change bytes")
            return _write_report(report, output_dir, path, started)

        if lief.PE.parse(mutated) is None:
            report.update(status="skipped_validation_failed", reason="Mutated PE failed LIEF parse")
            return _write_report(report, output_dir, path, started)

        output_path = output_dir / "exe" / f"{path.stem}_CR.exe"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(mutated)

        report.update(
            output_path=str(output_path),
            status="changed",
            reason=None,
            patch_count=len(selected),
            transform_counts=_transform_counts(selected),
            patches=[patch.to_manifest() for patch in selected],
            sha256_output=_sha256(mutated),
            size_original=len(original),
            size_output=len(mutated),
        )
        return _write_report(report, output_dir, path, started)

    except TimeoutError as exc:
        report.update(status="skipped_timeout", reason=str(exc))
        return _write_report(report, output_dir, path, started)
    except Exception as exc:
        logger.warning("Failed to process %s: %s", path, exc, exc_info=logger.isEnabledFor(logging.DEBUG))
        report.update(status="failed_exception", reason=str(exc))
        return _write_report(report, output_dir, path, started)


def _collect_candidates(
    *,
    path: Path,
    binary: Any,
    bits: int,
    raw_bytes: bytes,
    relocation_rvas: set[int],
    deadline: float,
    rng: random.Random,
) -> List[CandidatePatch]:
    project = angr.Project(
        str(path),
        auto_load_libs=False,
        load_options={"main_opts": {"base_addr": 0, "force_rebase": True}},
    )
    cfg = project.analyses.CFGFast(normalize=True)
    substituter = AtomicInstructionSubstituter(bits)
    reorderer = IntraBlockReorderer(bits)

    candidates: List[CandidatePatch] = []
    seen = set()
    for node in cfg.graph.nodes():
        _check_deadline(deadline)
        try:
            block = node.block
        except Exception:
            continue
        if block is None:
            continue

        block_insns = list(getattr(getattr(block, "capstone", None), "insns", []) or [])
        candidates.extend(
            reorderer.collect(
                block=block,
                binary=binary,
                raw_bytes=raw_bytes,
                relocation_rvas=relocation_rvas,
                rng=rng,
            )
        )

        for index, insn in enumerate(block_insns):
            rva = int(insn.address)
            size = int(insn.size)
            key = (rva, size)
            if key in seen:
                continue
            seen.add(key)

            file_offset = _rva_to_file_offset(binary, rva, size)
            if file_offset is None:
                continue

            candidate = substituter.collect(
                insn=insn,
                file_offset=file_offset,
                raw_bytes=raw_bytes,
                relocation_rvas=relocation_rvas,
                flags_dead_after=_flags_dead_after_instruction(block_insns, index),
            )
            if candidate is not None:
                candidates.append(candidate)

    preservation_reorderer = RegisterPreservationReorderer(bits)
    register_reassigner = RegisterReassigner(bits)
    for func in cfg.kb.functions.values():
        _check_deadline(deadline)
        if getattr(func, "is_simprocedure", False) or getattr(func, "is_plt", False):
            continue
        if getattr(func, "is_syscall", False):
            continue
        candidates.extend(
            preservation_reorderer.collect(
                func=func,
                binary=binary,
                raw_bytes=raw_bytes,
                relocation_rvas=relocation_rvas,
                rng=rng,
            )
        )
        candidates.extend(
            register_reassigner.collect(
                func=func,
                binary=binary,
                raw_bytes=raw_bytes,
                relocation_rvas=relocation_rvas,
                rng=rng,
            )
        )

    return sorted(candidates, key=lambda patch: patch.rva)


def _select_non_overlapping(
    candidates: Sequence[CandidatePatch],
    max_rewrites: int,
    rng: random.Random,
) -> List[CandidatePatch]:
    shuffled = list(candidates)
    rng.shuffle(shuffled)

    selected: List[CandidatePatch] = []
    used_ranges: List[range] = []
    if max_rewrites <= 0:
        return selected
    limit = max_rewrites

    for patch in shuffled:
        patch_range = range(patch.file_offset, patch.file_offset + len(patch.old_bytes))
        if any(_ranges_overlap(patch_range, used_range) for used_range in used_ranges):
            continue
        selected.append(patch)
        used_ranges.append(patch_range)
        if len(selected) >= limit:
            break

    return sorted(selected, key=lambda patch: patch.file_offset)


def _apply_patches(original: bytes, patches: Sequence[CandidatePatch]) -> bytes:
    mutated = bytearray(original)
    for patch in patches:
        start = patch.file_offset
        end = start + len(patch.old_bytes)
        if bytes(mutated[start:end]) != patch.old_bytes:
            raise ValueError(f"Patch precondition failed at file offset {start:#x}")
        mutated[start:end] = patch.new_bytes
    return bytes(mutated)


def _iter_inputs(args: argparse.Namespace) -> Iterable[Path]:
    if args.input:
        path = Path(args.input)
        if path.is_file():
            yield path
        return

    root = Path(args.input_dir)
    if not root.is_dir():
        return

    for path in sorted(root.rglob("*")):
        if path.is_file() and (path.suffix.lower() in SUPPORTED_SUFFIXES or not path.suffix):
            yield path


def _missing_dependencies() -> List[str]:
    missing = []
    if _ANGR_IMPORT_ERROR is not None:
        missing.append(f"angr ({_ANGR_IMPORT_ERROR})")
    if _LIEF_IMPORT_ERROR is not None:
        missing.append(f"lief ({_LIEF_IMPORT_ERROR})")
    if _CAPSTONE_IMPORT_ERROR is not None:
        missing.append(f"capstone ({_CAPSTONE_IMPORT_ERROR})")
    if _KEYSTONE_IMPORT_ERROR is not None:
        missing.append(f"keystone-engine ({_KEYSTONE_IMPORT_ERROR})")
    return missing


def _pe_bits_from_bytes(data: bytes) -> int:
    if len(data) < 0x40 or data[:2] != b"MZ":
        raise ValueError("not a valid MZ file")

    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    if pe_offset <= 0 or pe_offset + 6 > len(data) or data[pe_offset:pe_offset + 4] != b"PE\x00\x00":
        raise ValueError("not a valid PE file")

    machine = struct.unpack_from("<H", data, pe_offset + 4)[0]
    if machine == MACHINE_I386:
        return 32
    if machine == MACHINE_AMD64:
        return 64
    raise ValueError(f"unsupported PE machine type: {machine:#x}")


def _rva_to_file_offset(binary: Any, rva: int, size: int) -> Optional[int]:
    for section in binary.sections:
        try:
            virtual_start = int(section.virtual_address)
            raw_start = int(section.pointerto_raw_data)
            raw_size = int(section.sizeof_raw_data)
            virtual_size = max(int(section.virtual_size), raw_size)
            characteristics = int(section.characteristics)
        except Exception:
            continue

        if not _section_is_executable(characteristics):
            continue
        if virtual_start <= rva and rva + size <= virtual_start + virtual_size:
            offset = raw_start + (rva - virtual_start)
            if raw_start <= offset and offset + size <= raw_start + raw_size:
                return offset
    return None


def _section_is_executable(characteristics: int) -> bool:
    mem_execute = int(lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE)
    return bool(characteristics & mem_execute)


def _relocation_rvas(binary: Any) -> set[int]:
    rvas: set[int] = set()
    try:
        for relocation in binary.relocations:
            base = int(getattr(relocation, "virtual_address", 0))
            for entry in relocation.entries:
                position = getattr(entry, "position", None)
                if position is not None:
                    rvas.add(base + int(position))
                    continue

                address = getattr(entry, "address", None)
                if address is not None:
                    rvas.add(int(address))
    except Exception as exc:
        logger.debug("Could not enumerate relocation RVAs: %s", exc)
    return rvas


def _is_control_flow_instruction(insn: Any) -> bool:
    groups = set(getattr(insn, "groups", []) or [])
    if groups.intersection(CONTROL_FLOW_GROUPS):
        return True
    return insn.mnemonic in {"call", "jmp", "ret", "iret", "int", "syscall", "sysret"}


def _instruction_text(insn: Any) -> str:
    return f"{insn.mnemonic} {insn.op_str}".strip()


def _sequence_text(insns: Sequence[Any]) -> str:
    return "; ".join(_instruction_text(insn) for insn in insns)


def _normalize_instruction_text(insn: Any) -> str:
    return " ".join(_instruction_text(insn).lower().split())


def _is_nop_instruction(insn: Any) -> bool:
    return str(getattr(insn, "mnemonic", "") or "").lower() == "nop"


def _make_nop_padding(size: int) -> bytes:
    padding = bytearray()
    remaining = size
    for tile in _NOP_PADDING_TILES:
        while remaining >= len(tile):
            padding.extend(tile)
            remaining -= len(tile)
    if remaining != 0:
        raise ValueError("NOP padding tiling failed")
    return bytes(padding)


def _same_eflags_access(left: Any, right: Any) -> bool:
    return _eflags_access(left) == _eflags_access(right)


def _flags_dead_after_instruction(insns: Sequence[Any], index: int) -> bool:
    _, written_flags = _eflags_access(insns[index])
    pending = set(written_flags)
    if not pending:
        return False

    for next_insn in insns[index + 1:]:
        read_flags, write_flags = _eflags_access(next_insn)
        if read_flags.intersection(pending):
            return False
        pending.difference_update(write_flags)
        if not pending:
            return True
    return False


def _eflags_access(insn: Any) -> Tuple[Set[str], Set[str]]:
    mask = int(getattr(insn, "eflags", 0) or 0)
    reads: Set[str] = set()
    writes: Set[str] = set()
    for flag_name, spec in EFLAGS_SPECS.items():
        pseudo = f"eflags:{flag_name}"
        if mask & spec["read"]:
            reads.add(pseudo)
        if mask & spec["write"]:
            writes.add(pseudo)
    return reads, writes


def _has_pc_relative_operand(insn: Any) -> bool:
    for operand in getattr(insn, "operands", []) or []:
        if operand.type != X86_OP_MEM:
            continue
        base = int(getattr(operand.mem, "base", 0) or 0)
        if base in {X86_REG_EIP, X86_REG_RIP}:
            return True
    return False


def _has_complex_implicit_memory(insn: Any) -> bool:
    mnemonic = str(getattr(insn, "mnemonic", "") or "").lower()
    if mnemonic.startswith("rep") or mnemonic.startswith("lock"):
        return True
    return mnemonic in {
        "cmpsb", "cmpsw", "cmpsd", "cmpsq",
        "insb", "insw", "insd",
        "lodsb", "lodsw", "lodsd", "lodsq",
        "movsb", "movsw", "movsd", "movsq",
        "outsb", "outsw", "outsd",
        "scasb", "scasw", "scasd", "scasq",
        "stosb", "stosw", "stosd", "stosq",
        "enter", "leave",
    }


def _instruction_dependency(index: int, insn: Any) -> InstructionDependency:
    uses: Set[str] = set()
    defs: Set[str] = set()

    try:
        regs_read, regs_write = insn.regs_access()
    except Exception:
        regs_read = getattr(insn, "regs_read", []) or []
        regs_write = getattr(insn, "regs_write", []) or []

    for reg in regs_read:
        uses.add(_reg_name(insn, reg))
    for reg in regs_write:
        defs.add(_reg_name(insn, reg))

    memory_access = False
    for operand in getattr(insn, "operands", []) or []:
        if operand.type == X86_OP_MEM:
            memory_access = True
            base = int(getattr(operand.mem, "base", 0) or 0)
            index_reg = int(getattr(operand.mem, "index", 0) or 0)
            if base:
                uses.add(_reg_name(insn, base))
            if index_reg:
                uses.add(_reg_name(insn, index_reg))

    mnemonic = str(getattr(insn, "mnemonic", "") or "").lower()
    if mnemonic in {"push", "pop", "pusha", "pushad", "popa", "popad"}:
        memory_access = True

    flag_reads, flag_writes = _eflags_access(insn)
    uses.update(flag_reads)
    defs.update(flag_writes)
    return InstructionDependency(
        index=index,
        uses=uses,
        defs=defs,
        memory_access=memory_access,
        barrier=_is_exception_barrier(insn, memory_access),
    )


def _is_exception_barrier(insn: Any, memory_access: bool) -> bool:
    if memory_access:
        return True
    mnemonic = str(getattr(insn, "mnemonic", "") or "").lower()
    return mnemonic in TRAP_BARRIER_MNEMONICS


def _reg_name(insn: Any, reg_id: int) -> str:
    try:
        name = insn.reg_name(reg_id)
    except Exception:
        name = str(reg_id)
    return f"reg:{name}"


def _dependency_edges(instructions: Sequence[InstructionDependency]) -> Dict[int, Set[int]]:
    edges: Dict[int, Set[int]] = {info.index: set() for info in instructions}
    for left_index, left in enumerate(instructions):
        for right in instructions[left_index + 1:]:
            if _must_preserve_order(left, right):
                edges[left.index].add(right.index)
    return edges


def _must_preserve_order(left: InstructionDependency, right: InstructionDependency) -> bool:
    if left.barrier or right.barrier:
        return True
    if left.defs.intersection(right.uses):
        return True
    if left.uses.intersection(right.defs):
        return True
    if left.defs.intersection(right.defs):
        return True
    return left.memory_access and right.memory_access


def _random_topological_order(
    node_count: int,
    edges: Dict[int, Set[int]],
    original_order: Sequence[int],
    rng: random.Random,
) -> Optional[List[int]]:
    original = list(original_order)
    for _ in range(REORDER_ATTEMPTS):
        indegree = {index: 0 for index in range(node_count)}
        for sources in edges.values():
            for target in sources:
                indegree[target] += 1

        ready = [index for index, degree in indegree.items() if degree == 0]
        order: List[int] = []
        while ready:
            chosen = rng.choice(ready)
            ready.remove(chosen)
            order.append(chosen)
            for target in edges.get(chosen, set()):
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)

        if len(order) == node_count and order != original:
            return order
    return None


def _transform_counts(patches: Sequence[CandidatePatch]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for patch in patches:
        counts[patch.transform] = counts.get(patch.transform, 0) + 1
    return counts


def _ranges_overlap(left: range, right: range) -> bool:
    return left.start < right.stop and right.start < left.stop


def _sample_seed(base_seed: int, data: bytes) -> int:
    digest = hashlib.sha256(data).digest()
    sample_component = int.from_bytes(digest[:8], "little")
    return int(base_seed) ^ sample_component


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _check_deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise TimeoutError("per-sample CODE_RANDOMIZE timeout exceeded")


def _write_report(report: Dict[str, Any], output_dir: Path, input_path: Path, started: float) -> Dict[str, Any]:
    report["elapsed_sec"] = round(time.monotonic() - started, 3)
    report_path = output_dir / "json" / f"{input_path.stem}_CR.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


if __name__ == "__main__":
    raise SystemExit(main())
