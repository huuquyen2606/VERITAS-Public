from __future__ import annotations

import importlib.util
import logging
import random
import re
import struct
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import lief
except Exception as exc:  # pragma: no cover - dependency fallback
    lief = None
    _LIEF_IMPORT_ERROR = exc
else:
    _LIEF_IMPORT_ERROR = None

try:
    import pefile
except Exception as exc:  # pragma: no cover - dependency fallback
    pefile = None
    _PEFILE_IMPORT_ERROR = exc
else:
    _PEFILE_IMPORT_ERROR = None

try:
    from pwn import asm, context as pwn_context, p32, u32
except Exception as exc:  # pragma: no cover - dependency fallback
    asm = None
    pwn_context = None
    p32 = None
    u32 = None
    _PWN_IMPORT_ERROR = exc
else:
    _PWN_IMPORT_ERROR = None
    logging.getLogger("pwnlib").setLevel(logging.ERROR)
    logging.getLogger("pwnlib.asm").setLevel(logging.ERROR)


logger = logging.getLogger(__name__)

MACHINE_I386 = 0x014C
MACHINE_AMD64 = 0x8664
IMAGE_SCN_MEM_EXECUTE = 0x20000000
IMAGE_DIRECTORY_ENTRY_BOUND_IMPORT = 11

CALL_STUB_RE = re.compile(rb"(\xff\x15.{4})", re.DOTALL)
JUMP_STUB_RE = re.compile(rb"(\xff\x25.{4})", re.DOTALL)

ADDED_SECTION_NAME = b".added"
ADDED_SECTION_RAW_SIZE = 0x400
ADDED_SECTION_CHARACTERISTICS = 0xE0000020
JUMP_TABLE_ROW_SIZE = 20
CODE_STRING_OFFSET = 500


class StubType(Enum):
    CALL = "call"
    JUMP = "jump"


@dataclass(frozen=True)
class HijackCandidate:
    api_name: bytes
    call_rva: int
    stub_type: StubType


@dataclass
class AssemblyState:
    string_size: int = 0
    string_data: bytes = b""


@dataclass(frozen=True)
class HijackPlan:
    candidate: HijackCandidate
    dummy_api: bytes
    source: str


def bytecode_api_hijack(bytez: bytes, context: Optional[Dict[str, Any]] = None) -> bytes:
    """Patch one imported API callsite through a Tarallo-style trampoline."""
    original_bytes = bytes(bytez)

    try:
        if _PEFILE_IMPORT_ERROR is not None:
            logger.debug("BYTECODE_API_HIJACKING skipped: pefile unavailable: %s", _PEFILE_IMPORT_ERROR)
            return original_bytes
        if _LIEF_IMPORT_ERROR is not None:
            logger.debug("BYTECODE_API_HIJACKING skipped: lief unavailable: %s", _LIEF_IMPORT_ERROR)
            return original_bytes
        if _PWN_IMPORT_ERROR is not None:
            logger.debug("BYTECODE_API_HIJACKING skipped: pwntools unavailable: %s", _PWN_IMPORT_ERROR)
            return original_bytes

        pe = pefile.PE(data=original_bytes, fast_load=False)
        bits = _pe_bits(pe)
        _set_asm_context(bits)

        if _has_added_section(pe):
            logger.debug("BYTECODE_API_HIJACKING skipped: .added section already exists.")
            return original_bytes

        api_args = _load_tarallo_api_args()
        imported_apis = _imported_api_names(pe)
        injectable_apis = [api for api in api_args if api in imported_apis]
        if not injectable_apis:
            logger.debug("BYTECODE_API_HIJACKING skipped: no Tarallo dummy API is imported.")
            return original_bytes

        candidates = _retrieve_call_to_imported_functions(pe, bits)
        if not candidates:
            logger.debug("BYTECODE_API_HIJACKING skipped: no FF15/FF25 import stubs found.")
            return original_bytes

        plan = _choose_hijack_plan(
            candidates=candidates,
            injectable_apis=injectable_apis,
            imported_apis=imported_apis,
            api_args=api_args,
            context=context,
        )
        candidate = plan.candidate
        dummy_api = plan.dummy_api
        logger.debug(
            "BYTECODE_API_HIJACKING selected %s plan: hijack=%s inject=%s",
            plan.source,
            candidate.api_name.decode(errors="replace"),
            dummy_api.decode(errors="replace"),
        )
        data_to_inject, single_call_data_size = _single_api_hijacking_data()

        pe_data, jump_table_rva = _add_section(pe, ADDED_SECTION_NAME, ADDED_SECTION_RAW_SIZE)
        patched_pe = pefile.PE(data=pe_data, fast_load=False)

        iat_rva = _iat_entry_address_by_name(patched_pe, candidate.api_name)
        dummy_iat_rva = _iat_entry_address_by_name(patched_pe, dummy_api)
        if iat_rva is None or dummy_iat_rva is None:
            logger.debug("BYTECODE_API_HIJACKING skipped: selected API disappeared after section add.")
            return original_bytes

        iat_entries = [(iat_rva, data_to_inject, single_call_data_size)]
        injection = (
            _make_x86_bytecode(jump_table_rva, patched_pe, iat_entries, [dummy_api], api_args)
            if bits == 32
            else _make_x64_bytecode(jump_table_rva, patched_pe, iat_entries, [dummy_api], api_args)
        )
        if not injection or len(injection) > ADDED_SECTION_RAW_SIZE:
            logger.debug(
                "BYTECODE_API_HIJACKING skipped: injected code size %d exceeds .added raw size.",
                len(injection) if injection else 0,
            )
            return original_bytes

        _patch_callsite(patched_pe, candidate, jump_table_rva, bits)
        patched_pe.set_bytes_at_rva(jump_table_rva, injection)

        if _dynamic_base(patched_pe) and not _relocs_stripped(patched_pe):
            _patch_base_relocations(patched_pe, [candidate.call_rva + 2])

        if _has_old_style_delay_import(patched_pe):
            _repair_delay_import_in_memory(patched_pe, original_bytes)

        mutated = bytes(patched_pe.write())
        if not _validate_mutated_pe(mutated):
            logger.debug("BYTECODE_API_HIJACKING skipped: mutated PE failed validation.")
            return original_bytes

        return mutated

    except Exception as exc:
        logger.warning(
            "BYTECODE_API_HIJACKING failed; returning original bytes: %s",
            exc,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        return original_bytes


def _pe_bits(pe) -> int:
    machine = int(pe.FILE_HEADER.Machine)
    if machine == MACHINE_I386:
        return 32
    if machine == MACHINE_AMD64:
        return 64
    raise ValueError(f"unsupported PE machine type: {hex(machine)}")


def _set_asm_context(bits: int) -> None:
    if bits == 32:
        pwn_context.update(arch="i386", bits=32)
    else:
        pwn_context.update(arch="amd64", bits=64)
    pwn_context.log_level = "error"


def _align(value: int, alignment: int) -> int:
    if alignment <= 0:
        return value
    return ((value + alignment - 1) // alignment) * alignment


def _read_u32(data: bytes, signed: bool = False) -> int:
    if signed:
        return struct.unpack("<i", data)[0]
    return u32(data)


def _pack_rel32(value: int) -> bytes:
    if value < -(2**31) or value > (2**31 - 1):
        raise ValueError(f"relative displacement out of rel32 range: {value}")
    try:
        return p32(value, sign=True)
    except TypeError:
        return struct.pack("<i", value)


_TARALLO_API_ARGS_CACHE: Optional[Dict[bytes, Sequence[object]]] = None
_TARALLO_API_ARGS_LOAD_FAILED: bool = False


def _load_tarallo_api_args() -> Dict[bytes, Sequence[object]]:
    global _TARALLO_API_ARGS_CACHE, _TARALLO_API_ARGS_LOAD_FAILED

    if _TARALLO_API_ARGS_CACHE is not None:
        return _TARALLO_API_ARGS_CACHE
    if _TARALLO_API_ARGS_LOAD_FAILED:
        return {}    

    config_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "api_hijacking"
        / "config_api_args.py"
    )
    try:
        if not config_path.is_file():
            raise FileNotFoundError(f"API hijacking config_api_args.py not found at {config_path}")

        spec = importlib.util.spec_from_file_location("_tarallo_config_api_args", str(config_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load Tarallo API config from {config_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        api_args = getattr(module, "api_args", None)
        if not isinstance(api_args, dict):
            raise ValueError("Tarallo config_api_args.py does not define api_args dict")

        _TARALLO_API_ARGS_CACHE = api_args
        return api_args
    except Exception as exc:
        _TARALLO_API_ARGS_LOAD_FAILED = True
        logger.warning(
            "BYTECODE_API_HIJACKING disabled for this run: %s. "
            "Subsequent calls will return original bytes silently.",
            exc,
        )
        return {}


def _has_added_section(pe) -> bool:
    return any(section.Name.rstrip(b"\x00") == ADDED_SECTION_NAME for section in pe.sections)


def _imported_api_names(pe) -> set:
    names = set()
    for directory_name in ("DIRECTORY_ENTRY_IMPORT", "DIRECTORY_ENTRY_DELAY_IMPORT"):
        for entry in getattr(pe, directory_name, []) or []:
            for imp in getattr(entry, "imports", []) or []:
                if imp.name:
                    names.add(imp.name)
    return names


def _iat_entry_address_by_name(pe, function_name: bytes) -> Optional[int]:
    for directory_name in ("DIRECTORY_ENTRY_IMPORT", "DIRECTORY_ENTRY_DELAY_IMPORT"):
        for entry in getattr(pe, directory_name, []) or []:
            for imp in getattr(entry, "imports", []) or []:
                if imp.name == function_name:
                    return int(imp.address - pe.OPTIONAL_HEADER.ImageBase)
    return None


def _iat_entry_name_by_address(pe, iat_rva: int) -> Optional[bytes]:
    for directory_name in ("DIRECTORY_ENTRY_IMPORT", "DIRECTORY_ENTRY_DELAY_IMPORT"):
        for entry in getattr(pe, directory_name, []) or []:
            for imp in getattr(entry, "imports", []) or []:
                if imp.name and int(imp.address - pe.OPTIONAL_HEADER.ImageBase) == int(iat_rva):
                    return imp.name
    return None


def _retrieve_call_to_imported_functions(pe, bits: int) -> List[HijackCandidate]:
    mapped_image = pe.get_memory_mapped_image()
    candidates: List[HijackCandidate] = []

    for section in pe.sections:
        if not (int(section.Characteristics) & IMAGE_SCN_MEM_EXECUTE):
            continue

        start_rva = int(section.VirtualAddress)
        end_rva = start_rva + max(int(section.Misc_VirtualSize), int(section.SizeOfRawData))
        section_bytes = mapped_image[start_rva:end_rva]

        for match in CALL_STUB_RE.finditer(section_bytes):
            candidate = _candidate_from_stub(pe, bits, start_rva + match.start(), match.group(0), StubType.CALL)
            if candidate is not None:
                candidates.append(candidate)

        for match in JUMP_STUB_RE.finditer(section_bytes):
            candidate = _candidate_from_stub(pe, bits, start_rva + match.start(), match.group(0), StubType.JUMP)
            if candidate is not None:
                candidates.append(candidate)

    return candidates


def _candidate_from_stub(pe, bits: int, stub_rva: int, stub_bytes: bytes, stub_type: StubType) -> Optional[HijackCandidate]:
    if len(stub_bytes) != 6:
        return None

    if bits == 32:
        operand = _read_u32(stub_bytes[2:])
        iat_rva = int(operand - pe.OPTIONAL_HEADER.ImageBase)
    else:
        displacement = _read_u32(stub_bytes[2:], signed=True)
        iat_rva = int(stub_rva + 6 + displacement)

    api_name = _iat_entry_name_by_address(pe, iat_rva)
    if api_name is None:
        return None
    return HijackCandidate(api_name=api_name, call_rva=stub_rva, stub_type=stub_type)


def _single_api_hijacking_data() -> Tuple[bytes, int]:
    control_data = b"\x00\x00" + (2).to_bytes(2, "little")
    return control_data + b"\x00\xff", 2


def _choose_hijack_plan(
    candidates: Sequence[HijackCandidate],
    injectable_apis: Sequence[bytes],
    imported_apis: set,
    api_args: Dict[bytes, Sequence[object]],
    context: Optional[Dict[str, Any]],
) -> HijackPlan:
    sandbox_sequence = _context_ordered_api_sequence(context)
    sandbox_recorded_apis = set(sandbox_sequence)
    target_api_sequence = _context_target_api_sequence(context)
    target_api_set = set(target_api_sequence)

    if sandbox_sequence:
        candidate = _select_context_candidate(candidates, sandbox_sequence)
        dummy_api = _select_context_dummy_api(
            imported_apis=imported_apis,
            api_args=api_args,
            sandbox_sequence=sandbox_sequence,
            sandbox_recorded_apis=sandbox_recorded_apis,
            target_api_sequence=target_api_sequence,
            target_api_set=target_api_set,
        )
        if candidate is not None and dummy_api is not None:
            return HijackPlan(candidate=candidate, dummy_api=dummy_api, source="sandbox-guided")

        logger.debug(
            "BYTECODE_API_HIJACKING context planner fell back: candidate=%s dummy_api=%s",
            candidate is not None,
            dummy_api is not None,
        )

    return HijackPlan(
        candidate=random.choice(list(candidates)),
        dummy_api=random.choice(list(injectable_apis)),
        source="random-fallback",
    )


def _select_context_candidate(
    candidates: Sequence[HijackCandidate],
    sandbox_sequence: Sequence[bytes],
) -> Optional[HijackCandidate]:
    candidates_by_api: Dict[bytes, List[HijackCandidate]] = {}
    for candidate in sorted(candidates, key=lambda c: (c.api_name, c.call_rva)):
        candidates_by_api.setdefault(candidate.api_name, []).append(candidate)

    for api_name in sandbox_sequence:
        api_candidates = candidates_by_api.get(api_name)
        if api_candidates:
            return api_candidates[0]
    return None


def _select_context_dummy_api(
    imported_apis: set,
    api_args: Dict[bytes, Sequence[object]],
    sandbox_sequence: Sequence[bytes],
    sandbox_recorded_apis: set,
    target_api_sequence: Sequence[bytes],
    target_api_set: set,
) -> Optional[bytes]:
    feasible = set(imported_apis).intersection(api_args.keys()).intersection(sandbox_recorded_apis)
    if not feasible:
        return None

    if target_api_set:
        target_feasible = feasible.intersection(target_api_set)
        for api_name in target_api_sequence:
            if api_name in target_feasible:
                return api_name
        if target_feasible:
            return sorted(target_feasible)[0]

    for api_name in sandbox_sequence:
        if api_name in feasible:
            return api_name

    return sorted(feasible)[0]


def _context_ordered_api_sequence(context: Optional[Dict[str, Any]]) -> List[bytes]:
    if not isinstance(context, dict):
        return []

    for key in ("ordered_api_sequence", "api_sequence", "sandbox_api_sequence", "sandbox_apis"):
        if key in context:
            return _normalize_api_sequence(context.get(key))
    return []


def _context_target_api_sequence(context: Optional[Dict[str, Any]]) -> List[bytes]:
    if not isinstance(context, dict):
        return []

    for key in ("target_api_set", "target_apis", "preferred_injection_apis"):
        if key in context:
            return _normalize_api_sequence(context.get(key))
    return []


def _normalize_api_sequence(value: Any) -> List[bytes]:
    normalized: List[bytes] = []

    if value is None:
        return normalized

    if isinstance(value, (str, bytes, bytearray)):
        iterable = [value]
    elif isinstance(value, dict):
        direct_api_name = _normalize_api_name(value)
        iterable = [value] if direct_api_name is not None else list(value.keys())
    elif isinstance(value, set):
        iterable = sorted(value, key=lambda item: str(item))
    else:
        try:
            iterable = list(value)
        except TypeError:
            iterable = [value]

    for item in iterable:
        api_name = _normalize_api_name(item)
        if api_name is None:
            continue
        normalized.append(api_name)

    return normalized


def _normalize_api_name(value: Any) -> Optional[bytes]:
    if value is None:
        return None

    if isinstance(value, dict):
        for key in ("api", "api_name", "name", "function", "call"):
            if key in value:
                return _normalize_api_name(value.get(key))
        return None

    if isinstance(value, (tuple, list)):
        if not value:
            return None
        return _normalize_api_name(value[0])

    if isinstance(value, bytearray):
        raw = bytes(value)
    elif isinstance(value, bytes):
        raw = value
    elif isinstance(value, str):
        text = value.strip()
        if "!" in text:
            text = text.rsplit("!", 1)[1]
        raw = text.encode("utf-8", errors="ignore")
    else:
        return None

    raw = raw.strip().strip(b"\x00")
    if not raw or raw in {b"_PAD_", b"__exception__", b"__anomaly__"}:
        return None
    return raw


def _add_section(input_pe, section_name: bytes, raw_size: int, characteristics: int = ADDED_SECTION_CHARACTERISTICS):
    section_header_len = 0x28
    virtual_size = raw_size

    first_section_raw = int(input_pe.sections[0].PointerToRawData)
    new_section_offset = int(input_pe.sections[-1].get_file_offset() + section_header_len)
    if first_section_raw - new_section_offset < section_header_len:
        raise ValueError("not enough header padding to add .added section")

    bound_dir = input_pe.OPTIONAL_HEADER.DATA_DIRECTORY[IMAGE_DIRECTORY_ENTRY_BOUND_IMPORT]
    if int(bound_dir.Size) != 0:
        bound_import_table = input_pe.get_data(int(bound_dir.VirtualAddress), int(bound_dir.Size))
        input_pe.set_bytes_at_offset(int(bound_dir.VirtualAddress) + section_header_len, bound_import_table)
        bound_dir.VirtualAddress += section_header_len
        for structure in input_pe.__structures__:
            if structure.name in ("IMAGE_BOUND_IMPORT_DESCRIPTOR", "IMAGE_BOUND_FORWARDER_REF"):
                structure.set_file_offset(structure.get_file_offset() + section_header_len)

    overlay = input_pe.get_overlay()
    last_section = input_pe.sections[-1]
    file_alignment = int(input_pe.OPTIONAL_HEADER.FileAlignment)
    section_alignment = int(input_pe.OPTIONAL_HEADER.SectionAlignment)

    raw_size_aligned = _align(raw_size, file_alignment)
    virtual_size_aligned = _align(virtual_size, section_alignment)
    raw_offset = _align(int(last_section.PointerToRawData + last_section.SizeOfRawData), file_alignment)
    virtual_offset = _align(int(last_section.VirtualAddress + last_section.Misc_VirtualSize), section_alignment)

    if overlay is not None:
        raw_offset = _align(
            int(last_section.PointerToRawData + last_section.SizeOfRawData + len(overlay)),
            file_alignment,
        )

    section_name = section_name[:8].ljust(8, b"\x00")
    input_pe.set_bytes_at_offset(new_section_offset, section_name)
    input_pe.set_dword_at_offset(new_section_offset + 8, virtual_size_aligned)
    input_pe.set_dword_at_offset(new_section_offset + 12, virtual_offset)
    input_pe.set_dword_at_offset(new_section_offset + 16, raw_size_aligned)
    input_pe.set_dword_at_offset(new_section_offset + 20, raw_offset)
    input_pe.set_bytes_at_offset(new_section_offset + 24, b"\x00" * 12)
    input_pe.set_dword_at_offset(new_section_offset + 36, characteristics)

    input_pe.FILE_HEADER.NumberOfSections += 1
    input_pe.OPTIONAL_HEADER.SizeOfImage = virtual_offset + virtual_size_aligned

    if len(input_pe.__data__) < raw_offset:
        input_pe.__data__ += b"\x00" * (raw_offset - len(input_pe.__data__))
    input_pe.__data__ = input_pe.__data__[:raw_offset] + (b"\x00" * raw_size_aligned)
    input_pe.set_bytes_at_offset(raw_offset, b"\x00" * raw_size_aligned)

    return input_pe.write(), virtual_offset


def _patch_callsite(pe, candidate: HijackCandidate, jump_table_rva: int, bits: int) -> None:
    _set_asm_context(bits)
    instruction_size = 6

    if candidate.stub_type == StubType.CALL:
        displacement = int(jump_table_rva - candidate.call_rva - 5)
        new_code = b"\xe8" + _pack_rel32(displacement)
    else:
        displacement = int(jump_table_rva - candidate.call_rva)
        new_code = asm(f"jmp $+{hex(displacement)}")

    if len(new_code) > instruction_size:
        raise ValueError("patched callsite encoding is larger than original FF15/FF25 stub")
    pe.set_bytes_at_rva(candidate.call_rva, new_code.ljust(instruction_size, b"\x90"))


def _call_iat_entry_32(
    api_name: bytes,
    len_until_call: int,
    start_actual_code_rva: int,
    pe,
    api_args: Dict[bytes, Sequence[object]],
    state: AssemblyState,
) -> str:
    first_call_absolute_rva = start_actual_code_rva + len_until_call
    iat_entry_rva = _iat_entry_address_by_name(pe, api_name)
    if iat_entry_rva is None:
        raise ValueError(f"cannot inject missing API {api_name!r}")
    eip_offset = int(iat_entry_rva - first_call_absolute_rva)

    push_arguments = ""
    for arg in api_args[api_name]:
        if not isinstance(arg, bytes):
            push_arguments += f"push {arg}\n"
            continue

        string_pointer_offset = CODE_STRING_OFFSET + state.string_size
        push_arguments += "mov edx, ebx\n"
        push_arguments += f"add edx, {string_pointer_offset}\n"
        push_arguments += "push edx\n"
        state.string_size += len(arg)
        state.string_data += arg

    call_code = f"""
    push {eip_offset}
    pop ecx
    add ecx, ebx
    call dword ptr [ecx]
    """
    return push_arguments + call_code


def _make_x86_bytecode(
    jump_table_rva: int,
    pe,
    iat_entries: Sequence[Tuple[int, bytes, int]],
    api_injections: Sequence[bytes],
    api_args: Dict[bytes, Sequence[object]],
) -> bytes:
    state = AssemblyState()
    jmp_table_size = _compute_jmp_table_size(iat_entries)
    start_actual_code_rva = jump_table_rva + jmp_table_size

    _set_asm_context(32)
    prolog = """
    call lab
    lab:
    """
    len_until_call = len(asm(prolog))
    prolog += """
    push eax
    push ecx
    push edx
    push ebx
    push edi
    push esi
    push ebp
    add esp, 0x1c
    pop ebx
    pop esi
    pop ebp
    pop edi
    add edi, ebx
    sub esp, 0x2c
    """

    pre_injected_api_calls = """
    movzx eax, word ptr [ebp+ebx]
    cmp eax, esi
    je original_call
    movzx ecx, word ptr [ebp+ebx+2]
    add ecx, eax
    mov word ptr [ebp+ebx], cx
    add ebp, eax
    loop:
    movzx esi, byte ptr [ebp+ebx+4]
    cmp esi, 0xff
    je original_call
    """

    injected_api_calls = ""
    for i, api_name in enumerate(api_injections):
        api_call_code = _call_iat_entry_32(api_name, len_until_call, start_actual_code_rva, pe, api_args, state)
        injected_api_calls += f"cmp esi, {i}\n"
        injected_api_calls += f"jne $+{len(asm(api_call_code)) + 2}\n"
        injected_api_calls += api_call_code

    post_injected_api_calls = """
    inc ebp
    jmp loop
    """

    original_call_code = """
    original_call:
    mov eax, dword ptr [edi]
    mov dword ptr [esp+0x1c], eax
    pop ebp
    pop esi
    pop edi
    pop ebx
    pop edx
    pop ecx
    pop eax
    add esp, 0x10
    push dword ptr [esp-0x10]
    ret
    """

    injected_jump_table = _build_jump_table(iat_entries, jmp_table_size, start_actual_code_rva, len_until_call)
    injection = injected_jump_table + asm(
        prolog + pre_injected_api_calls + injected_api_calls + post_injected_api_calls + original_call_code
    )
    injection = injection.ljust(len_until_call + jmp_table_size + CODE_STRING_OFFSET, b"\x00")
    return injection + state.string_data


def _call_iat_entry_64(
    api_name: bytes,
    len_until_call: int,
    start_actual_code_rva: int,
    pe,
    api_args: Dict[bytes, Sequence[object]],
    state: AssemblyState,
    label_suffix: int,
) -> str:
    iat_entry_rva = _iat_entry_address_by_name(pe, api_name)
    if iat_entry_rva is None:
        raise ValueError(f"cannot inject missing API {api_name!r}")

    args_list = list(api_args[api_name])
    args_regs = ["rcx", "rdx", "r8", "r9"]
    code = ""

    if len(args_list) in [1, 2, 3, 4, 6, 8, 10, 12, 14]:
        code += f"""
        test rsp, 0x8
        jz no_extra_push{label_suffix}
        push rbx
        no_extra_push{label_suffix}:
        """
    else:
        code += f"""
        test rsp, 0x8
        jnz no_extra_push{label_suffix}
        push rbx
        no_extra_push{label_suffix}:
        """

    reordered_args = args_list[-4:][::-1] + args_list[:-4]
    for i, arg in enumerate(reordered_args):
        if not isinstance(arg, bytes):
            if i < 4:
                code += f"mov {args_regs[i]}, {arg}\n"
                if len(reordered_args) > 4:
                    code += "push 0x0\n"
            else:
                code += f"push {arg}\n"
            continue

        string_pointer_offset = CODE_STRING_OFFSET + state.string_size
        if i < 4:
            code += f"mov {args_regs[i]}, {string_pointer_offset}\n"
            code += f"add {args_regs[i]}, r14\n"
            if len(reordered_args) > 4:
                code += "push 0x0\n"
        else:
            code += "mov rax, r14\n"
            code += f"add rax, {string_pointer_offset}\n"
            code += "push rax\n"

        state.string_size += len(arg)
        state.string_data += arg

    iat_offset_from_lab = int(iat_entry_rva - (start_actual_code_rva + len_until_call))
    code += f"""
    mov rax, r14
    add rax, {iat_offset_from_lab}
    call qword ptr [rax]
    """

    if len(reordered_args) > 4:
        code += f"add rsp, {len(reordered_args) * 8}\n"

    code += f"""
    cmp qword ptr [rsp], rbx
    jne no_extra_pop{label_suffix}
    add rsp, 0x8
    no_extra_pop{label_suffix}:
    """
    return code


def _make_x64_bytecode(
    jump_table_rva: int,
    pe,
    iat_entries: Sequence[Tuple[int, bytes, int]],
    api_injections: Sequence[bytes],
    api_args: Dict[bytes, Sequence[object]],
) -> bytes:
    state = AssemblyState()
    jmp_table_size = _compute_jmp_table_size(iat_entries)
    start_actual_code_rva = jump_table_rva + jmp_table_size

    _set_asm_context(64)
    prolog = """
    call lab
    lab:
    """
    len_until_call = len(asm(prolog))
    prolog += """
    push rax
    push r10
    push r11
    push r15
    push r14
    push r13
    push rbx
    push rcx
    push rdx
    push r8
    push r9
    add rsp, 0x58
    pop r14
    pop r15
    pop r13
    pop rbx
    add rbx, r14
    sub rsp, 0x78
    """

    pre_injected_api_calls = """
    movzx rax, word ptr [r13+r14]
    cmp rax, r15
    je original_call
    movzx ecx, word ptr [r13+r14+2]
    add rcx, rax
    mov word ptr [r13+r14], cx
    add r13, rax
    loop:
    movzx r15, byte ptr [r13+r14+4]
    cmp r15, 0xff
    je original_call
    """

    injected_api_calls = ""
    for i, api_name in enumerate(api_injections):
        injected_api_calls += f"cmp r15, {i}\n"
        injected_api_calls += f"jne end_call_{i}\n"
        api_call_code = _call_iat_entry_64(
            api_name,
            len_until_call,
            start_actual_code_rva,
            pe,
            api_args,
            state,
            label_suffix=i,
        )
        injected_api_calls += api_call_code
        injected_api_calls += f"end_call_{i}:\n"

    post_injected_api_calls = """
    inc r13
    jmp loop
    """

    original_call_code = """
    original_call:
    mov rax, qword ptr [rbx]
    mov qword ptr [rsp+0x58], rax
    pop r9
    pop r8
    pop rdx
    pop rcx
    pop rbx
    pop r13
    pop r14
    pop r15
    pop r11
    pop r10
    pop rax
    add rsp, 0x20
    push qword ptr [rsp-0x20]
    ret
    """

    injected_jump_table = _build_jump_table(iat_entries, jmp_table_size, start_actual_code_rva, len_until_call)
    nop_sled = "nop\n" * 115
    injection = injected_jump_table + asm(
        prolog
        + pre_injected_api_calls
        + injected_api_calls
        + post_injected_api_calls
        + nop_sled
        + original_call_code
    )
    injection = injection.ljust(len_until_call + jmp_table_size + CODE_STRING_OFFSET, b"\x00")
    return injection + state.string_data


def _compute_jmp_table_size(iat_entries: Sequence[Tuple[int, bytes, int]]) -> int:
    return len(iat_entries) * JUMP_TABLE_ROW_SIZE + sum(len(entry[1]) for entry in iat_entries)


def _build_jump_table(
    iat_entries: Sequence[Tuple[int, bytes, int]],
    jmp_table_size: int,
    start_actual_code_rva: int,
    len_until_call: int,
) -> bytes:
    injected_jump_table = b""
    current_jmp_table_len = 0

    for iat_rva, api_data, single_call_data_size in iat_entries:
        current_jmp_table_len += JUMP_TABLE_ROW_SIZE
        start_data_offset = jmp_table_size - current_jmp_table_len + len_until_call
        current_jmp_table_len += len(api_data)

        first_call_rva = start_actual_code_rva + len_until_call
        iat_offset = int(iat_rva - first_call_rva)
        where_to_jump = int(jmp_table_size - current_jmp_table_len + len(api_data) + 5)

        fixed_size_code = b""
        fixed_size_code += asm(f"push {hex(iat_offset)}").ljust(5, b"\x90")
        fixed_size_code += asm(f"push {-start_data_offset}").ljust(5, b"\x90")
        fixed_size_code += asm(f"push {single_call_data_size}").ljust(5, b"\x90")
        fixed_size_code += asm(f"jmp $+{where_to_jump}").ljust(5, b"\x90")
        injected_jump_table += fixed_size_code + api_data

    return injected_jump_table


def _dynamic_base(pe) -> bool:
    return bool(getattr(pe.OPTIONAL_HEADER, "IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE", False))


def _relocs_stripped(pe) -> bool:
    return bool(getattr(pe.FILE_HEADER, "IMAGE_FILE_RELOCS_STRIPPED", False))


def _patch_base_relocations(pe, force_no_rebase_rvas: Sequence[int]) -> None:
    try:
        force_set = {int(rva) for rva in force_no_rebase_rvas}
        for relocation_base in getattr(pe, "DIRECTORY_ENTRY_BASERELOC", []) or []:
            for entry in getattr(relocation_base, "entries", []) or []:
                if int(entry.rva) in force_set:
                    entry.type = 0x0
    except Exception as exc:
        logger.debug("BYTECODE_API_HIJACKING relocation patch skipped: %s", exc)


def _has_old_style_delay_import(pe) -> bool:
    try:
        for delay_import in getattr(pe, "DIRECTORY_ENTRY_DELAY_IMPORT", []) or []:
            if int(delay_import.struct.grAttrs) == 0 and int(pe.FILE_HEADER.Machine) == MACHINE_I386:
                return True
    except Exception:
        return False
    return False


def _repair_delay_import_in_memory(pe, original_bytes: bytes) -> None:
    didata_size = 0
    didata_rva = 0

    for section in pe.sections:
        if section.Name.rstrip(b"\x00") == b".didata":
            didata_size = int(section.Misc)
            didata_rva = int(section.VirtualAddress)
            break

    if didata_rva == 0 or didata_size == 0:
        return

    original_pe = pefile.PE(data=original_bytes, fast_load=False)
    didata_offset_start = int(pe.get_offset_from_rva(didata_rva))
    didata_offset_end = int(pe.get_offset_from_rva(didata_rva + didata_size))
    original_didata = original_pe.__data__[didata_offset_start:didata_offset_end]
    pe.set_bytes_at_rva(didata_rva, original_didata)

    structures_to_delete = []
    for structure in pe.__structures__:
        if structure.name == "IMAGE_DELAY_IMPORT_DESCRIPTOR":
            structures_to_delete.append(structure)
        if (
            structure.name == "IMAGE_THUNK_DATA"
            and didata_offset_start <= structure.get_file_offset() < didata_offset_end
        ):
            structures_to_delete.append(structure)
    for structure in structures_to_delete:
        pe.__structures__.remove(structure)


def _validate_mutated_pe(mutated: bytes) -> bool:
    if not mutated:
        return False
    try:
        pefile.PE(data=mutated, fast_load=False)
    except Exception as exc:
        logger.debug("BYTECODE_API_HIJACKING pefile validation failed: %s", exc)
        return False

    try:
        return lief.PE.parse(mutated) is not None
    except Exception as exc:
        logger.debug("BYTECODE_API_HIJACKING LIEF validation failed: %s", exc)
        return False
