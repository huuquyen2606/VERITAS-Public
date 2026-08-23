"""Offline precomputation tool for CODE_TRANSLATION (#3).

This script is intentionally empirical: every PE candidate is sent through the
RetDec -> LLVM/OLLVM -> Clang/LLD pipeline. Failures are expected and are
recorded as JSON manifests; only fully validated outputs are written as
``*_CT.exe`` for online RL lookup. Missing artifacts are handled by the
environment as no-op/backend failures; the final MDP does not use action
masking.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import lief
import pefile


logger = logging.getLogger(__name__)

LLVM_VERSION_PATTERNS = (
    "opaque pointer",
    "opaque pointers",
    "invalid record",
    "invalid bitcode",
    "bitcode reader",
    "bitcode version",
    "producer",
    "reader",
    "llvm ir version",
    "llvm version",
    "invalid value for target triple",
)

IR_SYNTAX_PATTERNS = (
    "expected top-level entity",
    "expected value token",
    "expected instruction opcode",
    "invalid instruction",
    "use of undefined value",
    "unterminated",
    "expected type",
    "expected comma",
    "expected metadata",
)

UNRESOLVED_PATTERNS = (
    "undefined reference",
    "unresolved external",
    "symbol(s) not found",
    "undefined symbol",
    "could not find symbol",
    "ld.lld: error:",
)

PE32_MACHINE = 0x014C
PE64_MACHINE = 0x8664


@dataclass
class CommandResult:
    """Captured result for one external tool invocation."""

    stage: str
    args: List[str]
    returncode: Optional[int]
    stdout: str
    stderr: str
    elapsed_sec: float
    status: str
    timed_out: bool = False
    exception: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and self.exception is None

    def to_manifest(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "args": self.args,
            "returncode": self.returncode,
            "status": self.status,
            "timed_out": self.timed_out,
            "exception": self.exception,
            "elapsed_sec": round(self.elapsed_sec, 3),
            "stdout_tail": _tail(self.stdout),
            "stderr_tail": _tail(self.stderr),
        }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute CODE_TRANSLATION variants for PE files.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        help=(
            "Single PE file to process. Pair with GNU parallel (or any other "
            "fan-out tool) to precompute many samples concurrently — each "
            "invocation owns its own per-sample work subdirectory and writes "
            "outputs into the shared ``--output-dir``."
        ),
    )
    source.add_argument("--input-dir", help="Directory of PE files to process.")
    parser.add_argument(
        "--output-dir",
        default="data/precomputed/code-translation",
        help="Directory for *_CT.exe and *_CT.json outputs.",
    )
    parser.add_argument("--work-dir", default="/tmp/veritas_code_translation", help="Temporary working directory.")
    parser.add_argument("--retdec-bin", required=True, help="Path to RetDec LLVM IR lifting executable.")
    parser.add_argument("--llvm-dis-bin", default="llvm-dis", help="Path to llvm-dis for RetDec .bc fallback.")
    parser.add_argument("--opt-bin", default=None, help="Optional path to opt/OLLVM opt executable.")
    parser.add_argument("--clang-bin", required=True, help="Path to clang/OLLVM clang executable.")
    parser.add_argument(
        "--dlltool-bin",
        required=True,
        help=(
            "Default dlltool to use when a bit-specific override is not "
            "provided. Prefer ``llvm-dlltool`` here because it is bit-agnostic "
            "(driven by the ``-m`` flag); MinGW's ``x86_64-w64-mingw32-dlltool`` "
            "and ``i686-w64-mingw32-dlltool`` produce import libs whose machine "
            "type is fixed at build time and will collide with the other "
            "architecture at link."
        ),
    )
    parser.add_argument(
        "--dlltool-bin-32",
        default=None,
        help="Optional dlltool override used only for 32-bit (PE32) inputs.",
    )
    parser.add_argument(
        "--dlltool-bin-64",
        default=None,
        help="Optional dlltool override used only for 64-bit (PE32+) inputs.",
    )
    parser.add_argument(
        "--target",
        default="auto",
        choices=("auto", "x86_64-w64-windows-gnu", "i686-w64-windows-gnu"),
        help="Windows cross-compilation target triple.",
    )
    parser.add_argument(
        "--obf-flags",
        default="",
        help=(
            "Obfuscation flags passed to clang or opt. Empty by default so the "
            "stock LLVM toolchain works out-of-the-box; pass OLLVM-specific "
            "flags like '-mllvm -sub' only when --clang-bin points to an OLLVM "
            "build, otherwise stock clang will reject them."
        ),
    )
    parser.add_argument("--use-opt", action="store_true", help="Run opt over the lifted IR before clang.")
    parser.add_argument("--timeout-sec", type=int, default=900, help="Per-tool timeout in seconds.")
    parser.add_argument("--extensions", default=".exe", help="Comma-separated input extensions, or '*' for all files.")
    parser.add_argument("--keep-work-dir", action="store_true", help="Keep per-sample temporary artifacts.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    output_dir = Path(args.output_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    (output_dir / "exe").mkdir(parents=True, exist_ok=True)
    (output_dir / "json").mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        single = Path(args.input).resolve()
        if not single.is_file():
            logger.error("Input file not found: %s", single)
            return 1
        input_paths = [single]
    else:
        input_dir = Path(args.input_dir).resolve()
        input_paths = list(_iter_inputs(input_dir, args.extensions))
        if not input_paths:
            logger.error("No input files matched in %s", input_dir)
            return 1

    tool_versions = collect_tool_versions(args)
    successes = 0
    for path in input_paths:
        report = process_file(path, output_dir, work_dir, args, tool_versions)
        if report.get("status") == "changed":
            successes += 1
        logger.info("%s: %s", path.name, report.get("status"))

    logger.info("Finished CODE_TRANSLATION precompute: %d/%d succeeded", successes, len(input_paths))
    return 0


def process_file(
    path: Path,
    output_dir: Path,
    work_dir: Path,
    args: argparse.Namespace,
    tool_versions: Dict[str, Any],
) -> Dict[str, Any]:
    started = time.monotonic()
    sample_work_dir = work_dir / _safe_stem(path)
    output_path = output_dir / "exe" / f"{path.stem}_CT.exe"
    report: Dict[str, Any] = {
        "input_path": str(path),
        "output_path": None,
        "work_dir": str(sample_work_dir),
        "status": "failed_exception",
        "failed_stage": None,
        "reason": None,
        "elapsed_sec": 0.0,
        "commands": [],
        "tool_paths": {
            "retdec": args.retdec_bin,
            "llvm_dis": args.llvm_dis_bin,
            "opt": args.opt_bin,
            "clang": args.clang_bin,
            "dlltool": args.dlltool_bin,
        },
        "tool_versions": tool_versions,
        "imports": {},
        "artifacts": {},
        "validation": {},
    }

    try:
        original = path.read_bytes()
        report["sha256_original"] = _sha256(original)
        report["size_original"] = len(original)

        metadata = inspect_pe(path)
        report.update(metadata["report"])
        report["imports"] = metadata.get("imports", {})
        if not metadata["ok"]:
            report.update(
                status="skipped_parse_failed",
                failed_stage="parse",
                reason=metadata["reason"],
            )
            return write_report(report, output_dir, path, started)

        if sample_work_dir.exists():
            shutil.rmtree(sample_work_dir)
        sample_work_dir.mkdir(parents=True, exist_ok=True)

        lifted_ll = sample_work_dir / f"{path.stem}.ll"
        retdec_results = run_retdec_lift(path, lifted_ll, args, sample_work_dir)
        report["commands"].extend(result.to_manifest() for result in retdec_results)
        if not _nonempty_file(lifted_ll):
            failure = _first_failed(retdec_results)
            report.update(
                status=failure.status if failure else "skipped_lift_failed",
                failed_stage="lift",
                reason=_result_reason(failure) if failure else "RetDec did not produce a non-empty .ll file",
            )
            return write_report(report, output_dir, path, started)
        report["artifacts"]["lifted_ll"] = str(lifted_ll)

        current_ir = lifted_ll
        clang_obf_flags = shlex.split(args.obf_flags)
        if args.use_opt:
            transformed_ll = sample_work_dir / f"{path.stem}.opt.ll"
            opt_result = run_opt_transform(current_ir, transformed_ll, args, sample_work_dir)
            report["commands"].append(opt_result.to_manifest())
            if not opt_result.ok or not _nonempty_file(transformed_ll):
                report.update(
                    status=opt_result.status,
                    failed_stage="opt",
                    reason=_result_reason(opt_result),
                )
                return write_report(report, output_dir, path, started)
            current_ir = transformed_ll
            clang_obf_flags = []
            report["artifacts"]["transformed_ll"] = str(transformed_ll)

        dlltool_bin_for_arch = _select_dlltool(args, metadata["bits"])

        import_result = generate_import_libraries(
            pe_path=path,
            work_dir=sample_work_dir,
            dlltool_bin=dlltool_bin_for_arch,
            bits=metadata["bits"],
            timeout_sec=args.timeout_sec,
        )
        report["imports"] = import_result["manifest"]
        report["commands"].extend(result.to_manifest() for result in import_result["commands"])
        if import_result["failed_result"] is not None:
            failure = import_result["failed_result"]
            report.update(
                status=failure.status,
                failed_stage="import_libs",
                reason=_result_reason(failure),
            )
            return write_report(report, output_dir, path, started)

        import_exports: set = set()
        for dll_manifest in import_result["manifest"].get("dlls", []):
            for name in dll_manifest.get("named_imports") or ():
                if name:
                    import_exports.add(name)

        stubs_ll = generate_intrinsic_stubs(lifted_ll, sample_work_dir, import_exports)
        if stubs_ll is not None:
            report["artifacts"]["intrinsic_stubs_ll"] = str(stubs_ll)

        tmp_output = sample_work_dir / f"{path.stem}_CT.exe"
        target = _target_for(metadata["bits"], args.target)
        subsystem = metadata["report"]["subsystem_linker"]
        link_cmd = [
            args.clang_bin,
            "-target",
            target,
            "-fuse-ld=lld",
            "-nostartfiles",
            str(current_ir),
            *([str(stubs_ll)] if stubs_ll is not None else []),
            *[str(lib_path) for lib_path in import_result["libraries"]],
            *[f"-L{lib_dir}" for lib_dir in _mingw_library_dirs(metadata["bits"])],
            *clang_obf_flags,
            f"-Wl,-subsystem,{subsystem}",
            "-Wl,-e,entry_point",
            "-o",
            str(tmp_output),
        ]
        link_result = run_tool_safely(link_cmd, stage="link", timeout_sec=args.timeout_sec, cwd=sample_work_dir)
        report["commands"].append(link_result.to_manifest())
        if not link_result.ok or not _nonempty_file(tmp_output):
            report.update(
                status=link_result.status if link_result.status != "ok" else "skipped_link_failed",
                failed_stage="link",
                reason=_result_reason(link_result),
            )
            return write_report(report, output_dir, path, started)

        validation = validate_output(tmp_output)
        report["validation"] = validation
        if not validation["ok"]:
            report.update(
                status="skipped_validate_failed",
                failed_stage="validate",
                reason=validation["reason"],
            )
            return write_report(report, output_dir, path, started)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tmp_output, output_path)
        output_bytes = output_path.read_bytes()
        report.update(
            output_path=str(output_path),
            status="changed",
            failed_stage=None,
            reason=None,
            target_triple=target,
            sha256_output=_sha256(output_bytes),
            size_output=len(output_bytes),
        )
        return write_report(report, output_dir, path, started)

    except Exception as exc:
        logger.warning("CODE_TRANSLATION failed for %s: %s", path, exc, exc_info=logger.isEnabledFor(logging.DEBUG))
        report.update(status="failed_exception", failed_stage="exception", reason=str(exc))
        return write_report(report, output_dir, path, started)
    finally:
        if not args.keep_work_dir and sample_work_dir.exists():
            shutil.rmtree(sample_work_dir, ignore_errors=True)


def run_retdec_lift(
    input_path: Path,
    output_ll: Path,
    args: argparse.Namespace,
    cwd: Path,
) -> List[CommandResult]:
    """Lift a PE into textual LLVM IR via RetDec.

    RetDec 5.0 only supports ``--output-format ∈ {plain, json, json-human}``.
    Plain emits **C source** at the requested ``-o`` path, while LLVM bitcode
    is dropped at ``<stem>.bc`` as an intermediate artifact. The C source is
    not usable downstream, so we always:

      1. invoke RetDec to produce ``<stem>.bc`` (and discard the C source),
      2. run ``llvm-dis`` to convert that bitcode to textual ``.ll``.

    Returns the chain of CommandResults so the caller can surface failures.
    """
    cmd = [args.retdec_bin, str(input_path), "-o", str(output_ll), "--silent"]
    if output_ll.exists():
        output_ll.unlink()
    result = run_tool_safely(cmd, stage="lift", timeout_sec=args.timeout_sec, cwd=cwd)
    results: List[CommandResult] = [result]

    bc_path = output_ll.with_suffix(".bc")

    if output_ll.exists():
        try:
            output_ll.unlink()
        except OSError:
            pass

    if not _nonempty_file(bc_path):
        return results

    dis_cmd = [args.llvm_dis_bin, str(bc_path), "-o", str(output_ll)]
    dis_result = run_tool_safely(dis_cmd, stage="llvm_dis", timeout_sec=args.timeout_sec, cwd=cwd)
    results.append(dis_result)
    return results


_RETDEC_DECLARE_RE = re.compile(
    r"^\s*declare\s+"
    r"(?:(?:dso_local|local_unnamed_addr|unnamed_addr|noinline|nounwind|noreturn|inlinehint|alwaysinline|optnone|cold|naked|readnone|readonly|argmemonly|inaccessiblememonly|willreturn|mustprogress)\s+)*"
    r"(?P<ret>[^@]+?)\s+"
    r"@(?P<name>(?:\"[^\"]+\"|[\w.$]+))\s*"
    r"\((?P<args>[^)]*)\)",
    re.MULTILINE,
)


_LIKELY_WIN32_RE = re.compile(r"^[A-Z][\w]*$")


def _extract_target_lines(ll_text: str) -> List[str]:
    """Pull the ``target datalayout`` / ``target triple`` lines from RetDec IR."""
    lines: List[str] = []
    for line in ll_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("target datalayout") or stripped.startswith("target triple"):
            lines.append(stripped)
    return lines


_TYPE_DEF_RE = re.compile(r"^%[\w.$]+\s*=\s*type\s+.+$", re.MULTILINE)


_EXTERNAL_GLOBAL_RE = re.compile(
    r"^\s*@(?P<name>(?:\"[^\"]+\"|[\w.$]+))\s*=\s*external"
    r"(?:\s+(?:dso_local|local_unnamed_addr|unnamed_addr|thread_local|hidden|protected))*"
    r"\s+(?P<kind>global|constant)"
    r"\s+(?P<type>.+?)(?:,\s*align\s+\d+)?\s*$",
    re.MULTILINE,
)


def _extract_type_definitions(ll_text: str) -> List[str]:
    """Return all ``%X = type ...`` lines from the IR, in source order.

    Anonymous numbered types (``%0``, ``%1``, ...) must be emitted in the
    same order as the original module so cross-module structural matching
    works. We don't filter — copying the full table is cheap and avoids
    transitive-dependency analysis of nested struct fields.
    """
    return [m.group(0).strip() for m in _TYPE_DEF_RE.finditer(ll_text)]


def _zero_init_for(ret_type: str) -> str:
    """Return a literal LLVM IR value that satisfies a ``ret <ret_type>``.

    The mapping for floating-point scalars matters: LLVM rejects ``0.0`` for
    ``x86_fp80``, ``fp128``, and ``ppc_fp128`` because their representation
    width forces an explicit hex literal (the ``0xK...`` / ``0xL...`` /
    ``0xM...`` forms). The IEEE-binary scalars (``half``, ``bfloat``,
    ``float``, ``double``) accept the decimal ``0.0`` form.
    """
    rt = ret_type.strip()
    if rt == "void":
        return ""  
    if rt in {"half", "bfloat", "float", "double"}:
        return "0.0"
    if rt == "x86_fp80":
        return "0xK00000000000000000000"
    if rt == "fp128":
        return "0xL00000000000000000000000000000000"
    if rt == "ppc_fp128":
        return "0xM00000000000000000000000000000000"
    if re.fullmatch(r"i\d+", rt):
        return "0"
    if rt.endswith("*") or rt == "ptr":
        return "null"
    return "zeroinitializer"


def _split_top_level_commas(s: str) -> List[str]:
    """Split ``s`` on commas that are NOT inside brackets/parens/braces.

    LLVM IR types can contain top-level-looking commas inside ``<N x T>``,
    ``[N x T]``, ``{T1, T2}``, or function pointers ``T (T,T)*``. A naive
    ``s.split(',')`` mis-splits ``[8 x i16]**, i64`` into ``[8 x i16]**``
    plus ``i64`` (correct) but mangles ``{i32, i32}, i64`` into 3 pieces.
    Track depth across all four bracket families.
    """
    parts: List[str] = []
    depth = 0
    start = 0
    openers = "([<{"
    closers = ")]>}"
    for i, c in enumerate(s):
        if c in openers:
            depth += 1
        elif c in closers:
            depth = max(0, depth - 1)
        elif c == "," and depth == 0:
            parts.append(s[start:i])
            start = i + 1
    if start <= len(s):
        parts.append(s[start:])
    return [p.strip() for p in parts if p.strip()]


def _take_leading_type(arg: str) -> Optional[str]:
    """Pull the leading LLVM IR type off ``arg``, ignoring trailing attrs.

    ``arg`` is one element from ``_split_top_level_commas`` of a declare's
    argument list, e.g. ``"[8 x i16]**"`` or ``"i32 immarg"`` or just ``"i64"``.
    The return value is the type token alone — what we need to plug into a
    matching ``define`` signature.
    """
    s = arg.strip()
    if not s:
        return None
    depth = 0
    i = 0
    saw_non_space = False
    while i < len(s):
        c = s[i]
        if c in "([<{":
            depth += 1
            saw_non_space = True
        elif c in ")]>}":
            depth = max(0, depth - 1)
            saw_non_space = True
        elif c.isspace() and depth == 0 and saw_non_space:
            break
        else:
            saw_non_space = True
        i += 1
    return s[:i] or None


def _select_dlltool(args: argparse.Namespace, bits: int) -> str:
    """Return the dlltool binary to invoke for the given target bit width.

    Preference order:
      1. Explicit ``--dlltool-bin-32`` / ``--dlltool-bin-64`` override.
      2. The default ``--dlltool-bin``. Works only when it is the bit-
         agnostic ``llvm-dlltool`` (driven by ``-m``); a MinGW
         ``i686-w64-mingw32-dlltool`` would silently produce x86 import
         libs even when called for a 64-bit input.

    The script does not validate that the selected binary actually matches
    ``bits`` (we'd have to invoke it just to find out). We instead rely on
    the linker error to surface mismatches and let the user re-invoke with
    the right ``--dlltool-bin-NN`` override.
    """
    if bits == 32 and args.dlltool_bin_32:
        return args.dlltool_bin_32
    if bits == 64 and args.dlltool_bin_64:
        return args.dlltool_bin_64
    return args.dlltool_bin


def _is_likely_dll_import(name: str, import_exports: Optional[set]) -> bool:
    """Return True if ``name`` should be left to dlltool import libs.

    We skip emitting a stub when the symbol is plausibly a Win32 / WinRT API
    entry point, so the dlltool-generated import library can supply the real
    ``__imp_NAME`` thunk at link time. Heuristics, in order:

      1. C++ mangled names (``?...``, ``@@``) are NEVER import-lib targets;
         MSVC mangles only its own static C++ binaries.
      2. If the caller supplies an explicit ``import_exports`` set (the
         union of all named exports from the dlltool .def files), use it.
      3. Otherwise fall back to the heuristic that PascalCase identifiers
         are typically Win32 APIs.
    """
    if not name:
        return False
    if name.startswith("?") or "@@" in name or "$" in name:
        return False
    if import_exports is not None:
        return name in import_exports
    return bool(_LIKELY_WIN32_RE.match(name))


def generate_intrinsic_stubs(
    lifted_ll: Path,
    work_dir: Path,
    import_exports: Optional[set] = None,
) -> Optional[Path]:
    """Emit a sibling ``.ll`` with trivial definitions for every undefined
    ``declare`` in the lifted IR (except symbols that dlltool import libs
    will resolve).

    RetDec wraps a wide variety of unresolvable references as opaque
    ``declare`` lines:

      * x86 instructions it cannot translate (``__asm_paddd``, ``__asm_cpuid``,
        ``__asm_rep_stosq_memset``, ...).
      * MSVC C++ STL helpers when the original binary was statically linked
        (``?_Xlen@?$basic_string@...``, ``?_Syserror_map@std@@...``).
      * MSVC CRT internals (``__chkstk``, ``__scrt_acquire_startup_lock``,
        ``__vcrt_thread_attach``, ``_set_fmode``, ...).
      * Internal RetDec placeholders for unrecognised callees
        (``unknown_NNN``).

    All of these stay external in the lifted IR, so stock LLD fails the link
    with hundreds of "undefined symbol" errors. We emit a sibling
    ``<stem>.intrinsic_stubs.ll`` providing trivial definitions (return
    ``zeroinitializer`` / ``null`` / ``void``) with ``linkonce_odr`` linkage
    so any real implementation found later in the link line wins.

    Win32 / WinRT API names are skipped via ``_is_likely_dll_import`` so the
    dlltool-generated import libraries supply the proper ``__imp_*`` thunks.

    The resulting PE is structurally valid and links cleanly. Functional
    fidelity at the SIMD / static-CRT layer is not preserved — that's
    expected for action #3 (CODE_TRANSLATION); the env's reward pipeline
    assesses actual runtime behaviour and rolls back on functionality loss.
    """
    text = lifted_ll.read_text(errors="replace")
    seen: Dict[str, str] = {}
    for match in _RETDEC_DECLARE_RE.finditer(text):
        raw_name = match.group("name")
        if raw_name in seen:
            continue

        bare_name = raw_name[1:-1] if raw_name.startswith('"') and raw_name.endswith('"') else raw_name
        if _is_likely_dll_import(bare_name, import_exports):
            continue

        ret = match.group("ret").strip()
        args_raw = match.group("args").strip()
        if args_raw:
            arg_types: List[str] = []
            ok = True
            for arg in _split_top_level_commas(args_raw):
                arg_type = _take_leading_type(arg)
                if not arg_type:
                    ok = False
                    break
                arg_types.append(arg_type)
            if not ok:
                continue
        else:
            arg_types = []

        named_args = ", ".join(f"{t} %a{i}" for i, t in enumerate(arg_types))
        if ret == "void":
            body = "ret void"
        else:
            body = f"ret {ret} {_zero_init_for(ret)}"

        seen[raw_name] = f"define linkonce_odr {ret} @{raw_name}({named_args}) {{\n  {body}\n}}"

    global_stubs: Dict[str, str] = {}
    for match in _EXTERNAL_GLOBAL_RE.finditer(text):
        raw_name = match.group("name")
        if raw_name in global_stubs:
            continue
        bare_name = raw_name[1:-1] if raw_name.startswith('"') and raw_name.endswith('"') else raw_name
        if _is_likely_dll_import(bare_name, import_exports):
            continue
        kind = match.group("kind").strip()
        gtype = match.group("type").strip()
        gtype = gtype.rstrip(",").strip()
        global_stubs[raw_name] = (
            f"@{raw_name} = linkonce_odr {kind} {gtype} {_zero_init_for(gtype)}"
        )

    if not seen and not global_stubs:
        return None

    target_lines = _extract_target_lines(text)
    type_lines = _extract_type_definitions(text)

    chunks: List[str] = []
    if target_lines:
        chunks.append("\n".join(target_lines))
    if type_lines:
        chunks.append("\n".join(type_lines))
    if global_stubs:
        chunks.append("\n".join(global_stubs.values()))
    if seen:
        chunks.append("\n\n".join(seen.values()))

    stubs_path = work_dir / f"{lifted_ll.stem}.intrinsic_stubs.ll"
    stubs_path.write_text("\n\n".join(chunks) + "\n")
    return stubs_path


def run_opt_transform(input_ll: Path, output_ll: Path, args: argparse.Namespace, cwd: Path) -> CommandResult:
    if not args.opt_bin:
        return CommandResult(
            stage="opt",
            args=[],
            returncode=None,
            stdout="",
            stderr="--use-opt was set but --opt-bin is missing",
            elapsed_sec=0.0,
            status="skipped_tool_missing",
            exception="missing opt binary",
        )
    cmd = [
        args.opt_bin,
        *shlex.split(args.obf_flags),
        "-S",
        str(input_ll),
        "-o",
        str(output_ll),
    ]
    return run_tool_safely(cmd, stage="opt", timeout_sec=args.timeout_sec, cwd=cwd)


def run_tool_safely(
    args: Sequence[str],
    *,
    stage: str,
    timeout_sec: int,
    cwd: Optional[Path] = None,
) -> CommandResult:
    started = time.monotonic()
    args_list = [os.fspath(arg) for arg in args]
    try:
        completed = subprocess.run(
            args_list,
            cwd=os.fspath(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(1, int(timeout_sec)),
            check=False,
        )
        elapsed = time.monotonic() - started
        status = "ok" if completed.returncode == 0 else classify_failure(
            stage=stage,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        return CommandResult(
            stage=stage,
            args=args_list,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            elapsed_sec=elapsed,
            status=status,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        return CommandResult(
            stage=stage,
            args=args_list,
            returncode=None,
            stdout=_decode_process_output(exc.stdout),
            stderr=_decode_process_output(exc.stderr),
            elapsed_sec=elapsed,
            status="skipped_timeout",
            timed_out=True,
            exception=str(exc),
        )
    except FileNotFoundError as exc:
        elapsed = time.monotonic() - started
        return CommandResult(
            stage=stage,
            args=args_list,
            returncode=None,
            stdout="",
            stderr=str(exc),
            elapsed_sec=elapsed,
            status="skipped_tool_missing",
            exception=str(exc),
        )
    except Exception as exc:
        elapsed = time.monotonic() - started
        return CommandResult(
            stage=stage,
            args=args_list,
            returncode=None,
            stdout="",
            stderr=str(exc),
            elapsed_sec=elapsed,
            status="failed_exception",
            exception=str(exc),
        )


def classify_failure(stage: str, stdout: str, stderr: str) -> str:
    text = f"{stdout}\n{stderr}".lower()
    if _contains_any(text, LLVM_VERSION_PATTERNS):
        return "skipped_llvm_version_mismatch"
    if _contains_any(text, IR_SYNTAX_PATTERNS):
        return "skipped_ir_syntax_error"
    if stage == "lift":
        return "skipped_lift_failed"
    if stage == "opt":
        return "skipped_opt_failed"
    if stage == "llvm_dis":
        return "skipped_ir_syntax_error"
    if stage in {"dlltool", "link"}:
        return "skipped_link_failed"
    if _contains_any(text, UNRESOLVED_PATTERNS):
        return "skipped_link_failed"
    return "failed_exception"


def generate_import_libraries(
    *,
    pe_path: Path,
    work_dir: Path,
    dlltool_bin: str,
    bits: int,
    timeout_sec: int,
) -> Dict[str, Any]:
    imports = collect_imports(pe_path)
    import_dir = work_dir / "import_libs"
    import_dir.mkdir(parents=True, exist_ok=True)

    libraries: List[Path] = []
    commands: List[CommandResult] = []
    failed_result: Optional[CommandResult] = None
    dll_manifests: List[Dict[str, Any]] = []

    machine = "i386:x86-64" if bits == 64 else "i386"
    for dll_name, data in sorted(imports.items()):
        named_imports = sorted(data["names"])
        ordinal_imports = sorted(data["ordinals"])
        dll_manifest = {
            "dll": dll_name,
            "named_imports": named_imports,
            "ordinal_imports": ordinal_imports,
            "def_path": None,
            "library_path": None,
            "status": "skipped_no_named_imports",
        }

        if not named_imports:
            dll_manifests.append(dll_manifest)
            continue

        safe_name = _safe_lib_name(dll_name)
        def_path = import_dir / f"{safe_name}.def"
        lib_path = import_dir / f"lib{safe_name}.a"
        def_path.write_text(_render_def_file(dll_name, named_imports), encoding="utf-8")

        cmd = [
            dlltool_bin,
            "-m",
            machine,
            "--dllname",
            dll_name,
            "-d",
            str(def_path),
            "-l",
            str(lib_path),
        ]
        if bits == 32:
            cmd.insert(3, "--add-stdcall-alias")

        result = run_tool_safely(cmd, stage="dlltool", timeout_sec=timeout_sec, cwd=work_dir)
        commands.append(result)
        dll_manifest.update(def_path=str(def_path), library_path=str(lib_path), status=result.status)
        if not result.ok or not _nonempty_file(lib_path):
            failed_result = result
            dll_manifests.append(dll_manifest)
            break

        libraries.append(lib_path)
        dll_manifests.append(dll_manifest)

    manifest = {
        "dll_count": len(imports),
        "api_count": sum(len(data["names"]) for data in imports.values()),
        "ordinal_only_count": sum(len(data["ordinals"]) for data in imports.values()),
        "dlls": dll_manifests,
    }
    return {
        "libraries": libraries,
        "commands": commands,
        "failed_result": failed_result,
        "manifest": manifest,
    }


def collect_imports(pe_path: Path) -> Dict[str, Dict[str, set]]:
    imports: Dict[str, Dict[str, set]] = {}
    pe = pefile.PE(str(pe_path), fast_load=False)
    for directory_name in ("DIRECTORY_ENTRY_IMPORT", "DIRECTORY_ENTRY_DELAY_IMPORT"):
        for entry in getattr(pe, directory_name, []) or []:
            dll_name = _decode_bytes(getattr(entry, "dll", b"")).strip()
            if not dll_name:
                continue
            bucket = imports.setdefault(dll_name, {"names": set(), "ordinals": set()})
            for imp in getattr(entry, "imports", []) or []:
                if getattr(imp, "name", None):
                    name = _decode_bytes(imp.name).strip()
                    if _is_safe_export_name(name):
                        bucket["names"].add(name)
                elif getattr(imp, "ordinal", None) is not None:
                    bucket["ordinals"].add(int(imp.ordinal))
    return imports


def inspect_pe(path: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "arch": None,
        "bits": None,
        "target_triple": None,
        "subsystem": None,
        "subsystem_linker": None,
    }
    try:
        pe = pefile.PE(str(path), fast_load=False)
        machine = int(pe.FILE_HEADER.Machine)
        if machine == PE64_MACHINE:
            bits = 64
            arch = "pe64"
        elif machine == PE32_MACHINE:
            bits = 32
            arch = "pe32"
        else:
            return {"ok": False, "reason": f"unsupported PE machine 0x{machine:04x}", "report": report}

        subsystem_id = int(getattr(pe.OPTIONAL_HEADER, "Subsystem", 0))
        subsystem = {
            2: "windows",
            3: "console",
            9: "windowsce",
        }.get(subsystem_id, f"subsystem_{subsystem_id}")
        subsystem_linker = "console" if subsystem_id == 3 else "windows"
        report.update(
            arch=arch,
            bits=bits,
            subsystem=subsystem,
            subsystem_linker=subsystem_linker,
        )
        try:
            imports = _summarize_imports(collect_imports(path))
        except Exception as exc:
            imports = {"error": str(exc), "dll_count": 0, "api_count": 0, "ordinal_only_count": 0, "dlls": []}
        return {"ok": True, "reason": None, "report": report, "bits": bits, "imports": imports}
    except Exception as exc:
        return {"ok": False, "reason": str(exc), "report": report}


def _summarize_imports(imports: Dict[str, Dict[str, set]]) -> Dict[str, Any]:
    dlls = []
    for dll_name, data in sorted(imports.items()):
        dlls.append(
            {
                "dll": dll_name,
                "named_imports": sorted(data["names"]),
                "ordinal_imports": sorted(data["ordinals"]),
            }
        )
    return {
        "dll_count": len(imports),
        "api_count": sum(len(data["names"]) for data in imports.values()),
        "ordinal_only_count": sum(len(data["ordinals"]) for data in imports.values()),
        "dlls": dlls,
    }


def validate_output(path: Path) -> Dict[str, Any]:
    try:
        if not _nonempty_file(path):
            return {"ok": False, "reason": "output file missing or empty"}
        binary = lief.PE.parse(str(path))
        if binary is None:
            return {"ok": False, "reason": "LIEF returned None"}
        return {
            "ok": True,
            "reason": None,
            "output_size": path.stat().st_size,
            "section_count": len(binary.sections),
        }
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


def collect_tool_versions(args: argparse.Namespace) -> Dict[str, Any]:
    tools = {
        "retdec": args.retdec_bin,
        "opt": args.opt_bin,
        "clang": args.clang_bin,
        "dlltool": args.dlltool_bin,
    }
    versions: Dict[str, Any] = {}
    for name, path in tools.items():
        if not path:
            versions[name] = {"path": None, "status": "missing"}
            continue
        result = run_tool_safely([path, "--version"], stage=f"version_{name}", timeout_sec=10)
        versions[name] = result.to_manifest()
    return versions


def write_report(report: Dict[str, Any], output_dir: Path, input_path: Path, started: float) -> Dict[str, Any]:
    report["elapsed_sec"] = round(time.monotonic() - started, 3)
    report_path = output_dir / "json" / f"{input_path.stem}_CT.json"
    report["report_path"] = str(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _iter_inputs(input_dir: Path, extensions: str) -> Iterable[Path]:
    allowed = {item.strip().lower() for item in extensions.split(",") if item.strip()}
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        if "*" in allowed or path.suffix.lower() in allowed:
            yield path


def _target_for(bits: int, requested: str) -> str:
    if requested != "auto":
        return requested
    return "x86_64-w64-windows-gnu" if bits == 64 else "i686-w64-windows-gnu"


def _mingw_library_dirs(bits: int) -> List[Path]:
    arch = "x86_64-w64-mingw32" if bits == 64 else "i686-w64-mingw32"
    candidates: List[Path] = []
    gcc_root = Path("/usr/lib/gcc") / arch
    if gcc_root.is_dir():
        candidates.extend(sorted(gcc_root.glob("*-win32")))
        candidates.extend(sorted(gcc_root.glob("*-posix")))
    candidates.append(Path("/usr") / arch / "lib")
    return [path for path in candidates if path.is_dir()]


def _render_def_file(dll_name: str, exports: Sequence[str]) -> str:
    lines = [f"LIBRARY {dll_name}", "EXPORTS"]
    lines.extend(f"  {name}" for name in exports)
    return "\n".join(lines) + "\n"


def _is_safe_export_name(name: str) -> bool:
    return bool(name) and re.fullmatch(r"[A-Za-z_?$@][A-Za-z0-9_?$@.]*", name) is not None


def _safe_lib_name(dll_name: str) -> str:
    stem = Path(dll_name).stem.lower()
    return re.sub(r"[^a-z0-9_]+", "_", stem).strip("_") or "dll"


def _safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem) or "sample"


def _first_failed(results: Sequence[CommandResult]) -> Optional[CommandResult]:
    for result in results:
        if not result.ok:
            return result
    return results[-1] if results else None


def _result_reason(result: Optional[CommandResult]) -> Optional[str]:
    if result is None:
        return None
    if result.exception:
        return result.exception
    stderr = _tail(result.stderr, limit=1000).strip()
    stdout = _tail(result.stdout, limit=1000).strip()
    return stderr or stdout or f"{result.stage} returned {result.returncode}"


def _contains_any(text: str, patterns: Sequence[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def _nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _tail(text: Optional[str], limit: int = 4000) -> str:
    if not text:
        return ""
    return text[-limit:]


def _decode_process_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _decode_bytes(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
