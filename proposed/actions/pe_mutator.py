"""PE Mutator (LIEF-based PE file mutations)."""

from __future__ import annotations

import logging
import os
import pickle
import random
import struct
import tempfile
from typing import Any, Dict, List, Optional

import lief

try:
    from .cfg_nop_actions import CfgNopActions
    from .darkarmour_action import DarkarmourXORTransformAction
    from .packer_action import PackerTransform
    from .bytecode_api_hijacking_action import bytecode_api_hijack
except ImportError:
    from cfg_nop_actions import CfgNopActions
    from darkarmour_action import DarkarmourXORTransformAction
    from packer_action import PackerTransform
    from bytecode_api_hijacking_action import bytecode_api_hijack

logger = logging.getLogger(__name__)

_BENIGN_SECTION_NAMES: List[str] = [
    ".text", ".rdata", ".data", ".rsrc", ".reloc",
    ".pdata", ".tls", ".gfids", ".00cfg", ".idata",
    ".edata", ".bss", ".CRT", ".sxdata", ".rodata",
]


def _random_bytes(length: int) -> list:
    """Return a list of random ints [0,255]."""
    return [random.randint(0, 255) for _ in range(length)]


def _normalise_section_name(name: str) -> str:
    """Strip null padding and whitespace from a PE section name."""
    if name is None:
        return ""
    return name.rstrip("\x00").strip()


def _safe_build(pe: lief.PE.Binary, rebuild_imports: bool = False) -> Optional[bytes]:
    """Build PE bytes through tmpfs to avoid LIEF API hangs."""

    try:
        config = lief.PE.Builder.config_t()
        config.imports = rebuild_imports

        builder = lief.PE.Builder(pe, config)
        builder.build()

        ram_dir = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None
        fd, tmp_path = tempfile.mkstemp(suffix=".exe", dir=ram_dir)
        os.close(fd)

        try:
            builder.write(tmp_path)
            with open(tmp_path, "rb") as result:
                built_bytes = result.read()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError as cleanup_exc:
                logger.debug("Could not remove temporary PE build %s: %s", tmp_path, cleanup_exc)

        if not built_bytes:
            logger.warning("LIEF build produced empty output.")
            return None

        return built_bytes
    except Exception as exc:
        logger.warning("LIEF build failed: %s", exc)
        return None


class PEMutator:
    """Router for 19 PE mutation actions."""

    _DISPATCH = [
        "_dos_stub_perturb",          # 0
        "_content_shift",             # 1
        "_semantic_code_rewrite",     # 2
        "_code_translation",          # 3   precomputed macro-action
        "_overlay_append",            # 4
        "_imports_append",            # 5
        "_section_rename",            # 6
        "_section_add",               # 7
        "_section_append_bytes",      # 8
        "_remove_signature",          # 9
        "_remove_debug",              # 10
        "_break_checksum",            # 11
        "_packer_transform",          # 12  UPX adapter
        "_change_timestamp",          # 13
        "_xor_encryption",            # 14  Darkarmour adapter
        "_code_randomize",            # 15  precomputed macro-action
        "_bytecode_api_hijacking",    # 16
        "_cfg_edge_redivide",         # 17
        "_semantic_nop_inject",       # 18
    ]

    def __init__(
        self,
        xor_transform_action: Optional[DarkarmourXORTransformAction] = None,
        packer_transform_action: Optional[PackerTransform] = None,
        benign_content_dir: str = "data/benign_content",
    ) -> None:
        # External tool adapters.
        self.packer_transform_action = packer_transform_action or PackerTransform()
        self.xor_transform_action = xor_transform_action or DarkarmourXORTransformAction()
        self.cfg_nop_action = CfgNopActions(safe_build_fn=_safe_build)
        
        self.benign_bank = {
            "code_bytes": [],
            "data_bytes": [],
            "resource_bytes": [],
            "dos_stubs": [],
            "section_meta": [],
            "imports_dict": {}
        }
        self._agent_created_sections = set()
        
        self._load_benign_content(benign_content_dir)

    def _load_benign_content(self, benign_content_dir: str) -> None:
        """Safely load benign_bank.pkl."""
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        abs_benign_dir = os.path.join(repo_root, benign_content_dir)
        
        bank_path = os.path.join(abs_benign_dir, "benign_bank.pkl")
        if os.path.isfile(bank_path):
            try:
                with open(bank_path, "rb") as f:
                    bank_data = pickle.load(f)
                    # Update dictionary safely to keep default empty lists if missing
                    for k, v in bank_data.items():
                        if k in self.benign_bank:
                            self.benign_bank[k] = v
                logger.info("Loaded benign bank from %s", bank_path)
            except Exception as e:
                logger.warning("Failed to load benign bank: %s", e)
        else:
            logger.warning("Benign bank not found at %s. Will fallback to random/hardcoded bytes.", bank_path)

    def mutate(self, action_idx: int, bytez: bytes, context: Optional[Dict[str, Any]] = None) -> bytes:
        """Apply mutation ``action_idx`` to ``bytez``."""
        if not 0 <= action_idx < len(self._DISPATCH):
            logger.error("Invalid action index: %d", action_idx)
            return bytez

        method_name = self._DISPATCH[action_idx]
        method = getattr(self, method_name, None)
        if method is None:
            logger.debug("Action %d (%s) has no implementation; returning original.",
                         action_idx, method_name)
            return bytez

        try:
            if method_name in {"_code_translation", "_code_randomize", "_bytecode_api_hijacking"}:
                result = method(bytez, context=context)
            else:
                result = method(bytez)
            if result is None or len(result) == 0:
                logger.warning("Action %d (%s) returned empty; using original.",
                               action_idx, method_name)
                return bytez
            return result
        except Exception as exc:
            logger.warning("Action %d (%s) failed: %s; using original.",
                           action_idx, method_name, exc)
            return bytez

    def _dos_stub_perturb(self, bytez: bytes) -> bytes:
        """Overwrite unused bytes in the DOS stub with random data."""
        bytez_mut = bytearray(bytez)

        # Read e_lfanew (little-endian uint32 at offset 0x3C)
        if len(bytez_mut) < 0x40:
            return bytez
        e_lfanew = struct.unpack_from("<I", bytez_mut, 0x3C)[0]
        if e_lfanew < 0x40 or e_lfanew >= len(bytez_mut):
            return bytez

        # Pick a full benign DOS stub if available
        if self.benign_bank["dos_stubs"]:
            benign_stub = random.choice(self.benign_bank["dos_stubs"])
            # The dos stub contains both regions.
            # We must be careful not to overwrite MZ (0x00-0x01) and e_lfanew (0x3C-0x3F)
            
            # Truncate or pad benign stub to match our DOS region (up to e_lfanew)
            benign_stub = benign_stub[:e_lfanew]
            if len(benign_stub) < e_lfanew:
                benign_stub += [0] * (e_lfanew - len(benign_stub))
                
            # Replace region 1 [0x02, 0x3C)
            for i in range(0x02, 0x3C):
                if i < len(benign_stub):
                    bytez_mut[i] = benign_stub[i]
                    
            # Replace region 2 [0x40, e_lfanew)
            for i in range(0x40, e_lfanew):
                if i < len(benign_stub):
                    bytez_mut[i] = benign_stub[i]
        else:
            # Fallback to random bytes
            safe_regions = []
            if 0x3C > 0x02:
                safe_regions.append((0x02, 0x3C))
            if e_lfanew > 0x40:
                safe_regions.append((0x40, e_lfanew))
    
            if not safe_regions:
                return bytez
    
            # Pick one region and overwrite a random chunk
            start, end = random.choice(safe_regions)
            region_len = end - start
            overwrite_len = random.randint(1, min(region_len, 32))
            offset = random.randint(start, end - overwrite_len)
            for i in range(overwrite_len):
                bytez_mut[offset + i] = random.randint(0, 255)

        return bytes(bytez_mut)

    def _overlay_append(self, bytez: bytes) -> bytes:
        """Append random bytes to the PE overlay."""
        if len(bytez) < 2 or bytez[:2] != b"MZ":
            logger.debug("OVERLAY_APPEND skipped: input does not look like a PE.")
            return bytez

        append_size = random.randint(32, 1024)
        pool = self.benign_bank["data_bytes"] + self.benign_bank["resource_bytes"]
        
        if pool:
            # Pick a random benign chunk
            chunk = random.choice(pool)
            # Truncate or pad to append_size
            if len(chunk) > append_size:
                start_idx = random.randint(0, len(chunk) - append_size)
                noise = chunk[start_idx:start_idx + append_size]
            else:
                noise = chunk
            return bytez + bytes(noise)
            
        noise = bytes([random.randint(0, 255) for _ in range(append_size)])
        return bytez + noise

    def _section_rename(self, bytez: bytes) -> bytes:
        """Rename a random section to a benign-looking name."""
        pe = lief.PE.parse(bytez)
        if pe is None:
            return bytez

        sections = list(pe.sections)
        if not sections:
            return bytez

        benign_names = _BENIGN_SECTION_NAMES
        if self.benign_bank["section_meta"]:
            benign_names = list(set(meta[0] for meta in self.benign_bank["section_meta"]))
            
        target = random.choice(sections)
        candidates = [n for n in benign_names if n != target.name]
        if not candidates:
            candidates = benign_names
        target.name = random.choice(candidates)

        result = _safe_build(pe)
        return result if result else bytez

    def _section_add(self, bytez: bytes) -> bytes:
        """Add a new inert section with random content."""
        pe = lief.PE.parse(bytez)
        if pe is None:
            return bytez

        section = lief.PE.Section()
        
        sec_name = random.choice(_BENIGN_SECTION_NAMES)
        sec_chars = (lief.PE.Section.CHARACTERISTICS.MEM_READ | 
                     lief.PE.Section.CHARACTERISTICS.CNT_INITIALIZED_DATA)
                     
        # and LIEF Python's bitwise-and between an int and a CHARACTERISTICS
        # enum value isn't reliable across versions. Use int() on both sides.
        mem_execute = int(lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE)

        if self.benign_bank["section_meta"]:
            # Filter for non-executable sections
            inert_metas = [meta for meta in self.benign_bank["section_meta"]
                           if not (int(meta[1]) & mem_execute)]
            if inert_metas:
                sec_name, sec_chars = random.choice(inert_metas)

        existing_names = {s.name for s in pe.sections}
        if sec_name in existing_names:
            sec_name = ".s%d" % random.randint(0, 99)

        section.name = sec_name
        section.characteristics = sec_chars

        content_pool = []
        if int(sec_chars) & mem_execute:
            content_pool = self.benign_bank["code_bytes"]
        elif sec_name == ".rsrc":
            content_pool = self.benign_bank["resource_bytes"]
        else:
            content_pool = self.benign_bank["data_bytes"]

        if content_pool:
            chunk = random.choice(content_pool)
            content_size = random.randint(64, 512)
            if len(chunk) > content_size:
                start_idx = random.randint(0, len(chunk) - content_size)
                section.content = chunk[start_idx:start_idx + content_size]
            else:
                section.content = chunk
        else:
            content_size = random.randint(64, 512)
            section.content = _random_bytes(content_size)

        pe.add_section(section)

        result = _safe_build(pe)
        if result:
            # Normalise the cached name so later lookups in _content_shift
            # match even if LIEF's Section.name accessor returns a value with
            # null-padding or trailing whitespace on a different version.
            self._agent_created_sections.add(_normalise_section_name(section.name))
            return result
        return bytez

    def _section_append_bytes(self, bytez: bytes) -> bytes:
        """Append random bytes into the slack space of an existing section."""
        pe = lief.PE.parse(bytez)
        if pe is None:
            return bytez

        # Find sections with slack space (raw_size > virtual_size)
        candidates = []
        for s in pe.sections:
            slack = s.sizeof_raw_data - s.virtual_size
            if slack > 0:
                candidates.append((s, slack))

        if not candidates:
            # Fallback: just overlay-append
            return self._overlay_append(bytez)

        target_section, slack = random.choice(candidates)
        fill_size = random.randint(1, min(slack, 256))

        # Extend the virtual size to consume slack
        current_content = list(target_section.content)
        # Pad content up to virtual_size if needed, then append
        while len(current_content) < target_section.virtual_size:
            current_content.append(0)
            
        if self.benign_bank["data_bytes"]:
            chunk = random.choice(self.benign_bank["data_bytes"])
            if len(chunk) > fill_size:
                start_idx = random.randint(0, len(chunk) - fill_size)
                noise = chunk[start_idx:start_idx + fill_size]
            else:
                noise = chunk
            current_content.extend(noise)
        else:
            current_content.extend(_random_bytes(fill_size))
            
        target_section.content = current_content
        target_section.virtual_size = len(current_content)

        result = _safe_build(pe)
        return result if result else bytez

    def _remove_signature(self, bytez: bytes) -> bytes:
        """Clear the CERTIFICATE_TABLE data directory (Authenticode)."""
        pe = lief.PE.parse(bytez)
        if pe is None:
            return bytez

        cert_dir = pe.data_directory(lief.PE.DataDirectory.TYPES.CERTIFICATE_TABLE)
        if cert_dir is None:
            return bytez

        cert_dir.rva = 0
        cert_dir.size = 0

        result = _safe_build(pe)
        return result if result else bytez

    def _remove_debug(self, bytez: bytes) -> bytes:
        """Clear the DEBUG data directory."""
        pe = lief.PE.parse(bytez)
        if pe is None:
            return bytez

        debug_dir = pe.data_directory(lief.PE.DataDirectory.TYPES.DEBUG_DIR)
        if debug_dir is None:
            return bytez

        debug_dir.rva = 0
        debug_dir.size = 0

        if hasattr(pe, 'debug') and pe.debug:
            for dbg in pe.debug:
                if hasattr(dbg, 'sizeof_data'):
                    dbg.sizeof_data = 0

        result = _safe_build(pe)
        return result if result else bytez

    def _break_checksum(self, bytez: bytes) -> bytes:
        """Set OptionalHeader.CheckSum to 0."""
        pe = lief.PE.parse(bytez)
        if pe is None:
            return bytez

        pe.optional_header.checksum = 0

        result = _safe_build(pe)
        return result if result else bytez

    def _change_timestamp(self, bytez: bytes) -> bytes:
        """Randomize IMAGE_FILE_HEADER.TimeDateStamp."""
        pe = lief.PE.parse(bytez)
        if pe is None:
            return bytez

        # randint() is inclusive on both ends, so use the last second of 2024
        # (not 2025-01-01 midnight) to keep the range strictly within the
        # documented [2010, 2024] window.
        ts_min = 1262304000   # 2010-01-01 00:00:00 UTC
        ts_max = 1735689599   # 2024-12-31 23:59:59 UTC
        pe.header.time_date_stamps = random.randint(ts_min, ts_max)

        result = _safe_build(pe)
        return result if result else bytez

    def _content_shift(self, bytez: bytes) -> bytes:
        """Shift execution away from EntryPoint into an agent-created section."""
        original_bytes = bytes(bytez)

        try:
            if not self._agent_created_sections:
                logger.debug("CONTENT_SHIFT skipped: no agent-created sections recorded.")
                return original_bytes

            pe = lief.PE.parse(original_bytes)
            if pe is None:
                logger.debug("CONTENT_SHIFT skipped: LIEF returned no PE binary.")
                return original_bytes

            data_directory_ranges = []
            for directory_type in lief.PE.DataDirectory.TYPES:
                try:
                    directory = pe.data_directory(directory_type)
                except Exception:
                    continue
                if directory and directory.rva != 0 and directory.size != 0:
                    data_directory_ranges.append((directory.rva, directory.rva + directory.size))

            safe_sections = []
            mem_execute = int(lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE)
            for section in pe.sections:
                if _normalise_section_name(section.name) not in self._agent_created_sections:
                    continue

                if int(section.characteristics) & mem_execute:
                    continue

                content = list(section.content)
                if len(content) < 64:
                    continue

                section_start = section.virtual_address
                section_end = section_start + max(section.virtual_size, len(content))
                overlaps_data_directory = any(
                    not (section_end <= directory_start or section_start >= directory_end)
                    for directory_start, directory_end in data_directory_ranges
                )
                if overlaps_data_directory:
                    continue

                safe_sections.append((section, content))

            if not safe_sections:
                logger.debug("CONTENT_SHIFT skipped: no safe agent-created section found.")
                return original_bytes

            target, target_content = random.choice(safe_sections)
            max_shift = min(256, max(16, len(target_content) // 4))
            shift_size = random.randint(16, max_shift)

            if self.benign_bank["data_bytes"]:
                chunk = list(random.choice(self.benign_bank["data_bytes"]))
                padding = chunk[:shift_size]
                if len(padding) < shift_size:
                    padding.extend([0] * (shift_size - len(padding)))
            else:
                padding = _random_bytes(shift_size)

            shifted_content = padding + target_content
            target.content = shifted_content
            target.virtual_size = len(shifted_content)

            result = _safe_build(pe)
            return result if result else original_bytes

        except Exception as exc:
            logger.warning(
                "CONTENT_SHIFT failed; returning original bytes: %s",
                exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            return original_bytes

    def _semantic_code_rewrite(self, bytez: bytes) -> bytes:
        """Re-assemble instructions without length changes."""
        original_bytes = bytes(bytez)

        try:
            try:
                from capstone import Cs, CS_ARCH_X86, CS_MODE_32, CS_MODE_64
            except ImportError as exc:
                logger.debug("SEMANTIC_CODE_REWRITE skipped: capstone unavailable: %s", exc)
                return original_bytes

            pe = lief.PE.parse(original_bytes)
            if pe is None:
                logger.debug("SEMANTIC_CODE_REWRITE skipped: LIEF returned no PE binary.")
                return original_bytes

            is_pe64 = pe.optional_header.magic == lief.PE.PE_TYPE.PE32_PLUS
            mode = CS_MODE_64 if is_pe64 else CS_MODE_32

            # Same-size rewrite rules.  Avoid substitutions such as
            # xor reg,reg <-> sub reg,reg and add reg,0 <-> or reg,0 because
            # they differ on auxiliary-flag behavior on real x86 CPUs.
            rewrite_pairs_32 = [
                (b"\x85\xc0", b"\x21\xc0"),  # test eax,eax <-> and eax,eax
                (b"\x85\xc9", b"\x21\xc9"),  # test ecx,ecx <-> and ecx,ecx
                (b"\x85\xd2", b"\x21\xd2"),  # test edx,edx <-> and edx,edx
                (b"\x85\xdb", b"\x21\xdb"),  # test ebx,ebx <-> and ebx,ebx
                (b"\x09\xc0", b"\x21\xc0"),  # or eax,eax   <-> and eax,eax
                (b"\x09\xc9", b"\x21\xc9"),  # or ecx,ecx   <-> and ecx,ecx
                (b"\x09\xd2", b"\x21\xd2"),  # or edx,edx   <-> and edx,edx
                (b"\x09\xdb", b"\x21\xdb"),  # or ebx,ebx   <-> and ebx,ebx
                (b"\x89\xc0", b"\x90\x90"),  # mov eax,eax  -> nop; nop
                (b"\x89\xc9", b"\x90\x90"),  # mov ecx,ecx  -> nop; nop
                (b"\x89\xd2", b"\x90\x90"),  # mov edx,edx  -> nop; nop
                (b"\x89\xdb", b"\x90\x90"),  # mov ebx,ebx  -> nop; nop
            ]
            rewrite_pairs_64 = [
                (b"\x48\x85\xc0", b"\x48\x21\xc0"),  # test rax,rax <-> and rax,rax
                (b"\x48\x85\xc9", b"\x48\x21\xc9"),  # test rcx,rcx <-> and rcx,rcx
                (b"\x48\x85\xd2", b"\x48\x21\xd2"),  # test rdx,rdx <-> and rdx,rdx
                (b"\x48\x85\xdb", b"\x48\x21\xdb"),  # test rbx,rbx <-> and rbx,rbx
                (b"\x48\x09\xc0", b"\x48\x21\xc0"),  # or rax,rax   <-> and rax,rax
                (b"\x48\x09\xc9", b"\x48\x21\xc9"),  # or rcx,rcx   <-> and rcx,rcx
                (b"\x48\x09\xd2", b"\x48\x21\xd2"),  # or rdx,rdx   <-> and rdx,rdx
                (b"\x48\x09\xdb", b"\x48\x21\xdb"),  # or rbx,rbx   <-> and rbx,rbx
                (b"\x48\x89\xc0", b"\x90\x90\x90"),  # mov rax,rax  -> nop; nop; nop
                (b"\x48\x89\xc9", b"\x90\x90\x90"),  # mov rcx,rcx  -> nop; nop; nop
                (b"\x48\x89\xd2", b"\x90\x90\x90"),  # mov rdx,rdx  -> nop; nop; nop
                (b"\x48\x89\xdb", b"\x90\x90\x90"),  # mov rbx,rbx  -> nop; nop; nop
            ]

            rewrite_pairs = rewrite_pairs_64 if is_pe64 else rewrite_pairs_32
            rewrite_map = {}
            for original, replacement in rewrite_pairs:
                if len(original) != len(replacement):
                    continue
                rewrite_map.setdefault(original, set()).add(replacement)
                rewrite_map.setdefault(replacement, set()).add(original)

            if not rewrite_map:
                return original_bytes

            md = Cs(CS_ARCH_X86, mode)
            md.skipdata = True

            mem_execute = int(lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE)
            mutated_bytes = bytearray(original_bytes)
            section_candidates = []
            for section in pe.sections:
                if not (int(section.characteristics) & mem_execute):
                    continue

                content = bytearray(section.content)
                if not content:
                    continue

                raw_start = int(section.pointerto_raw_data)
                raw_size = min(
                    len(content),
                    int(section.sizeof_raw_data),
                    max(0, len(original_bytes) - raw_start),
                )
                if raw_start < 0 or raw_size <= 0:
                    continue

                candidates = []
                for insn in md.disasm(bytes(content[:raw_size]), section.virtual_address):
                    insn_bytes = bytes(insn.bytes)
                    replacements = rewrite_map.get(insn_bytes)
                    if not replacements:
                        continue

                    offset = int(insn.address) - int(section.virtual_address)
                    if offset < 0 or offset + len(insn_bytes) > raw_size:
                        continue
                    if bytes(content[offset:offset + len(insn_bytes)]) != insn_bytes:
                        continue

                    candidates.append((offset, insn_bytes, tuple(replacements)))

                if candidates:
                    section_candidates.append((raw_start, candidates))

            if not section_candidates:
                logger.debug("SEMANTIC_CODE_REWRITE skipped: no safe same-size rewrite candidates.")
                return original_bytes

            rewrites_left = random.randint(
                1,
                min(50, sum(len(candidates) for _, candidates in section_candidates)),
            )
            changed = False

            random.shuffle(section_candidates)
            for raw_start, candidates in section_candidates:
                if rewrites_left <= 0:
                    break

                random.shuffle(candidates)
                used_ranges = []
                for offset, expected, replacements in candidates:
                    if rewrites_left <= 0:
                        break
                    end = offset + len(expected)
                    if any(not (end <= start or offset >= stop) for start, stop in used_ranges):
                        continue
                    raw_offset = raw_start + offset
                    raw_end = raw_offset + len(expected)
                    if raw_offset < 0 or raw_end > len(mutated_bytes):
                        continue
                    if bytes(mutated_bytes[raw_offset:raw_end]) != expected:
                        continue

                    replacement = random.choice(replacements)
                    if replacement == expected:
                        continue

                    mutated_bytes[raw_offset:raw_end] = replacement
                    used_ranges.append((offset, end))
                    rewrites_left -= 1
                    changed = True

            if not changed:
                logger.debug("SEMANTIC_CODE_REWRITE skipped: all candidates became stale.")
                return original_bytes

            # Same-size raw patching should preserve the PE layout. Parse once
            # after mutation to catch malformed edge cases without rebuilding.
            if lief.PE.parse(bytes(mutated_bytes)) is None:
                logger.debug("SEMANTIC_CODE_REWRITE skipped: mutated PE did not parse.")
                return original_bytes

            return bytes(mutated_bytes)

        except Exception as exc:
            logger.warning(
                "SEMANTIC_CODE_REWRITE failed; returning original bytes: %s",
                exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            return original_bytes

    def _code_translation(self, bytez: bytes, context: Optional[Dict[str, Any]] = None) -> bytes:
        """Replace the PE with a translated version from context."""
        if not isinstance(context, dict):
            logger.debug("CODE_TRANSLATION skipped: missing context.")
            return bytez

        precomputed_path = context.get("code_translation_path")
        if not precomputed_path:
            logger.debug("CODE_TRANSLATION skipped: context has no code_translation_path.")
            return bytez

        try:
            path = os.path.abspath(os.fspath(precomputed_path))
        except (TypeError, ValueError):
            logger.debug("CODE_TRANSLATION skipped: invalid code_translation_path=%r.", precomputed_path)
            return bytez

        if not os.path.isfile(path):
            logger.debug("CODE_TRANSLATION skipped: precomputed file not found at %s.", path)
            return bytez

        try:
            with open(path, "rb") as sample:
                mutated = sample.read()

            if not mutated:
                logger.warning("CODE_TRANSLATION precomputed file is empty: %s", path)
                return bytez

            if lief.PE.parse(mutated) is None:
                logger.debug("CODE_TRANSLATION skipped: precomputed PE failed LIEF parse.")
                return bytez

            return mutated
        except Exception as exc:
            logger.warning("CODE_TRANSLATION failed to load %s: %s; returning original.", path, exc)
            return bytez

    def _imports_append(self, bytez: bytes) -> bytes:
        """Add one benign, unused DLL/API import entry to the PE IAT."""
        original_bytes = bytes(bytez)
        benign_imports = self.benign_bank["imports_dict"]

        try:
            if not original_bytes:
                logger.debug("IMPORTS_APPEND skipped: empty PE bytes.")
                return original_bytes

            if not benign_imports:
                logger.debug("IMPORTS_APPEND using built-in fallback benign imports.")
                benign_imports = {
                    "ADVAPI32.dll": ["OpenProcessToken", "RegOpenKeyExA", "RegQueryValueExA"],
                    "KERNEL32.dll": ["GetTickCount", "GetCurrentProcessId", "GetSystemTime"],
                    "USER32.dll": ["MessageBoxA", "GetDesktopWindow", "LoadIconA"],
                }

            import_candidates = []
            for dll_name, api_names in benign_imports.items():
                if not isinstance(dll_name, str) or not dll_name.strip():
                    continue
                valid_api_names = [
                    api_name for api_name in api_names
                    if isinstance(api_name, str) and api_name.strip()
                ]
                if valid_api_names:
                    import_candidates.append((dll_name.strip(), valid_api_names))

            if not import_candidates:
                logger.warning("IMPORTS_APPEND skipped: no valid DLL/API candidates.")
                return original_bytes

            dll_name, api_names = random.choice(import_candidates)
            api_name = random.choice(api_names).strip()

            binary = lief.PE.parse(original_bytes)
            if binary is None:
                logger.debug("IMPORTS_APPEND skipped: LIEF returned no PE binary.")
                return original_bytes

            selected_dll = dll_name.lower()
            library = None
            for imported_library in binary.imports:
                imported_name = getattr(imported_library, "name", "")
                if imported_name and imported_name.lower() == selected_dll:
                    library = imported_library
                    break

            if library is None:
                library = binary.add_import(dll_name)

            existing_entries = {
                entry.name for entry in library.entries
                if getattr(entry, "name", None)
            }
            if api_name not in existing_entries:
                library.add_entry(api_name)

            mutated_bytes = _safe_build(binary, rebuild_imports=True)

            if not mutated_bytes:
                logger.warning("IMPORTS_APPEND produced empty output; returning original bytes.")
                return original_bytes

            return mutated_bytes

        except Exception as exc:
            logger.warning(
                "IMPORTS_APPEND failed; returning original bytes: %s",
                exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            return original_bytes

    def _packer_transform(self, bytez: bytes) -> bytes:
        """Pack PE via the bundled UPX executable."""
        try:
            return self.packer_transform_action.apply_bytes(bytez)
        except Exception as exc:
            logger.warning("packer_transform failed: %s; returning original.", exc)
            return bytez

    def _xor_encryption(self, bytez: bytes) -> bytes:
        """XOR-encrypt payload and wrap in a darkarmour JMP-loader stub."""
        try:
            return self.xor_transform_action.apply(bytez)
        except Exception as exc:
            logger.warning("darkarmour XOR transform failed: %s; returning original.", exc)
            return bytez

    def _code_randomize(self, bytez: bytes, context: Optional[Dict[str, Any]] = None) -> bytes:
        """Replace the PE with an obfuscated version from context."""
        if not isinstance(context, dict):
            logger.debug("CODE_RANDOMIZE skipped: missing context.")
            return bytez

        precomputed_path = context.get("precomputed_path")
        if not precomputed_path:
            logger.debug("CODE_RANDOMIZE skipped: context has no precomputed_path.")
            return bytez

        try:
            path = os.path.abspath(os.fspath(precomputed_path))
        except (TypeError, ValueError):
            logger.debug("CODE_RANDOMIZE skipped: invalid precomputed_path=%r.", precomputed_path)
            return bytez

        if not os.path.isfile(path):
            logger.debug("CODE_RANDOMIZE skipped: precomputed file not found at %s.", path)
            return bytez

        try:
            if lief.PE.parse(path) is None:
                logger.debug("CODE_RANDOMIZE skipped: precomputed PE failed LIEF parse.")
                return bytez

            with open(path, "rb") as sample:
                mutated = sample.read()

            if not mutated:
                logger.warning("CODE_RANDOMIZE precomputed file is empty: %s", path)
                return bytez

            return mutated
        except Exception as exc:
            logger.warning("CODE_RANDOMIZE failed to load %s: %s; returning original.", path, exc)
            return bytez

    def _bytecode_api_hijacking(self, bytez: bytes, context: Optional[Dict[str, Any]] = None) -> bytes:
        """Patch one import callsite through a sandbox-guided Tarallo trampoline."""
        return bytecode_api_hijack(bytez, context=context)

    def _cfg_edge_redivide(self, bytez: bytes) -> bytes:
        """Split a call-site edge through a bridge block."""
        try:
            return self.cfg_nop_action.cfg_edge_redivide(bytez)
        except Exception as exc:
            logger.warning("cfg_edge_redivide failed: %s; returning original.", exc)
            return bytez

    def _semantic_nop_inject(self, bytez: bytes) -> bytes:
        """Inject semantic NOP templates into bridge code or slack."""
        try:
            return self.cfg_nop_action.semantic_nop_inject(bytez)
        except Exception as exc:
            logger.warning("semantic_nop_inject failed: %s; returning original.", exc)
            return bytez