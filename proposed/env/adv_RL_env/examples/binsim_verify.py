import os
import sys
import json
import argparse
import re
import concurrent.futures
from triton import (
    TritonContext,
    ARCH,
    Instruction,
    AST_REPRESENTATION,
    MODE,
    MemoryAccess,
)
from z3 import Solver, unsat, sat, parse_smt2_string

sys.setrecursionlimit(100000)

# ==========================================
# --- CONFIGURATION SETTINGS ---
# ==========================================

# Concurrency & Hardware Limits
MAX_CONCURRENT_SLICES = 14  # Number of pairs processed simultaneously
Z3_THREADS = 4  # Threads per Z3 solver instance (14 * 4 = 56 cores utilized)
Z3_TIMEOUT_MS = 55000  # Maximum time (ms) Z3 will spend proving a single slice

# Pipeline I/O Defaults
DEFAULT_SLICES_DIR = "binsim_slices"
REPORT_FILENAME = "binsim_final_report.json"

# Crypto Detection Thresholds
CRYPTO_THRESHOLD_XOR = 200
CRYPTO_THRESHOLD_SHL = 200

# Strict Syscall Parameter Mapping
SYSCALL_ARG_COUNTS = {
    0x42: 11,
    0x30: 6,
    0x0C: 1,
    0x112: 9,
    0xFE: 9,
    0x13: 1,
    0x1D: 7,
    0x0F: 3,
    0x5D: 6,
    0x3F: 2,
    0x15: 6,
    0x37: 5,
    0x10A: 10,
    0x4D: 5,
    0x12A: 2,
    0x2F: 8,
    0x23: 4,
    0x35: 8,
    0xCE: 2,
    0x114: 5,
    0xD5: 2,
    0x07: 10,
}


# ==========================================
# PHASE 1: TRITON LIFTER (In-Memory)
# ==========================================
def is_crypto_formula(formula_str: str) -> bool:
    return (
        formula_str.count("bvxor") > CRYPTO_THRESHOLD_XOR
        or formula_str.count("bvshl") > CRYPTO_THRESHOLD_SHL
    )


def lift_to_smt(slice_path: str):
    if not os.path.exists(slice_path):
        return None

    with open(slice_path, "r") as f:
        slice_data = json.load(f)

    instructions = slice_data.get("instruction_segment", [])
    args_concrete = slice_data.get("syscall_args_concrete", [])
    memory_log = slice_data.get("memory_log", [])
    target_id = slice_data.get("target_syscall", {}).get("id", 0)

    if not instructions:
        if not args_concrete:
            return "(_ bv0 32)"
        combined = f"FALLBACK_ARG_{len(args_concrete) - 1}"
        for j in reversed(range(len(args_concrete) - 1)):
            combined = f"(concat FALLBACK_ARG_{j} {combined})"
        return combined

    ctx = TritonContext(ARCH.X86)
    ctx.setAstRepresentationMode(AST_REPRESENTATION.SMT)
    ctx.setMode(MODE.ONLY_ON_SYMBOLIZED, False)
    ctx.setMode(MODE.AST_OPTIMIZATIONS, True)
    ctx.setMode(MODE.CONSTANT_FOLDING, False)
    ctx.setMode(MODE.ALIGNED_MEMORY, True)

    first_regs = instructions[0].get("regs", {}) if instructions else {}
    initial_edx_val = (
        int(first_regs.get("edx", "0x0"), 16) if "edx" in first_regs else 0
    )

    prior_retvals = set()
    for ret in slice_data.get("prior_syscall_returns", []):
        val = int(ret.get("retval", "0x0"), 16)
        if val != 0:
            prior_retvals.add(val)

    REG_NAMES = ["eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi"]
    for r_name in REG_NAMES:
        reg_obj = getattr(ctx.registers, r_name)
        val = int(first_regs.get(r_name, "0x0"), 16)
        ctx.setConcreteRegisterValue(reg_obj, val)
        if val in prior_retvals and val != 0:
            ctx.symbolizeRegister(reg_obj, f"REG_{r_name.upper()}_TAINT")
        elif r_name in ["eax", "ebx", "ecx", "edx"]:
            ctx.symbolizeRegister(reg_obj, f"REG_{r_name.upper()}")

    for mem_entry in memory_log:
        try:
            addr, size, value = (
                int(mem_entry["addr"], 16),
                int(mem_entry["size"]),
                int(mem_entry["value"], 16),
            )
            ctx.setConcreteMemoryValue(MemoryAccess(addr, size), value)
            if mem_entry.get("type") == "R" and value in prior_retvals and value != 0:
                ctx.symbolizeMemory(MemoryAccess(addr, size), f"MEM_{hex(addr)}")
        except:
            continue

    for block in instructions:
        try:
            pc, raw_bytes = int(block["pc"], 16), bytes.fromhex(block["bytes"])
            block_size = block.get("size", len(raw_bytes))
            offset = 0
            while offset < min(block_size, len(raw_bytes)):
                chunk = raw_bytes[offset : offset + 15]
                if not chunk:
                    break
                inst = Instruction(pc + offset, chunk)
                ctx.processing(inst)
                disas = inst.getDisassembly().lower()
                offset += inst.getSize() if inst.getSize() > 0 else 1
                if "sysenter" in disas or "int 0x2e" in disas:
                    break
        except:
            continue

    arg_count = SYSCALL_ARG_COUNTS.get(target_id, 4)
    arg_formulas = []
    for j in range(arg_count):
        addr = initial_edx_val + 4 + (j * 4)
        try:
            arg_formulas.append(str(ctx.getMemoryAst(MemoryAccess(addr, 4))))
        except:
            arg_formulas.append("(_ bv0 32)")

    if not arg_formulas:
        return "(_ bv0 32)"
    combined = arg_formulas[-1]
    for f in reversed(arg_formulas[:-1]):
        combined = f"(concat {f} {combined})"
    formula_str = combined

    if is_crypto_formula(formula_str):
        return "[CRYPTO_DETECTED]"

    unique_slots = list(
        dict.fromkeys(re.compile(r"MEM_0x[0-9a-f]+").findall(formula_str))
    )
    for i, slot in enumerate(unique_slots):
        formula_str = formula_str.replace(slot, f"NORM_SLOT_{i}")

    unique_refs = list(dict.fromkeys(re.compile(r"ref!\d+").findall(formula_str)))
    for i, ref in enumerate(unique_refs):
        formula_str = formula_str.replace(ref, f"REF_{i}")

    return formula_str


# ==========================================
# PHASE 2: Z3 PROVER (In-Memory)
# ==========================================
def load_expr(formula_str):
    known_vars = set(
        re.findall(
            r"(REG_[A-Z0-9_]+|SymVar_[0-9]+|NORM_SLOT_[0-9]+|ref![0-9]+|k![0-9]+|INIT_MEM_[a-f0-9x]+|FALLBACK_ARG_[0-9]+)",
            formula_str,
        )
    )
    decls_map = {v: "(_ BitVec 32)" for v in known_vars}
    current_probe_size = "32"

    while True:
        decls = "".join(
            f"(declare-const {v} {sort})\n" for v, sort in sorted(decls_map.items())
        )
        smt2_script = f"(set-logic QF_BV)\n{decls}\n(declare-const __probe__ (_ BitVec {current_probe_size}))\n(assert (= __probe__ {formula_str}))\n(check-sat)"
        try:
            assertions = parse_smt2_string(smt2_script)
            if not assertions:
                return None
            return assertions[0].arg(1)
        except Exception as e:
            error_msg = str(e)
            if "are incompatible" in error_msg:
                sizes = re.findall(r"\(_ BitVec (\d+)\)", error_msg)
                if len(sizes) >= 2:
                    target_size = (
                        sizes[0] if sizes[0] != current_probe_size else sizes[1]
                    )
                    if target_size == current_probe_size:
                        return None
                    current_probe_size = target_size
                    continue
            match_const = re.search(
                r"unknown constant (ref!\d+|k!\d+|[a-zA-Z0-9_!]+)", error_msg
            )
            if match_const:
                decls_map[match_const.group(1)] = "(_ BitVec 32)"
                continue
            return None


def evaluate_pair(orig_raw, adv_raw):
    if not orig_raw or not adv_raw:
        return 0.5, "Missing Trace"
    if "[CRYPTO_DETECTED]" in orig_raw or "[CRYPTO_DETECTED]" in adv_raw:
        return 0.7, "Crypto Approx"
    if orig_raw == adv_raw:
        return 1.0, "Exact String Match"

    expr_orig = load_expr(orig_raw)
    expr_adv = load_expr(adv_raw)
    if expr_orig is None or expr_adv is None or expr_orig.sort() != expr_adv.sort():
        return 0.5, "Parse Error"

    s1 = Solver()
    s1.set("timeout", Z3_TIMEOUT_MS)
    s1.set("threads", Z3_THREADS)
    s1.add(expr_orig != expr_adv)

    try:
        if s1.check() == unsat:
            return 1.0, "Mathematically Identical"

        s2 = Solver()
        s2.set("timeout", Z3_TIMEOUT_MS)
        s2.set("threads", Z3_THREADS)
        s2.add(expr_orig == expr_adv)
        if s2.check() == sat:
            return 0.5, "Conditional Equivalence"
        return 0.0, "Different"
    except:
        return 0.5, "Solver Timeout"


# ==========================================
# PHASE 3: MULTI-THREADED WORKER
# ==========================================
def process_slice_task(i, orig_path, adv_path):
    """Worker function to lift and evaluate a single slice pair."""
    try:
        formula_orig = lift_to_smt(orig_path)
        formula_adv = lift_to_smt(adv_path)
        score, reason = evaluate_pair(formula_orig, formula_adv)
        return {"slice_index": i, "score": score, "reason": reason}
    except Exception as e:
        return {
            "slice_index": i,
            "score": 0.5,
            "reason": f"Unhandled Exception: {str(e)[:40]}",
        }


# ==========================================
# MAIN ORCHESTRATION & REPORTING
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Multi-Threaded BinSim Z3 Verifier")
    parser.add_argument(
        "--slices-dir",
        default=DEFAULT_SLICES_DIR,
        help="Directory containing malware pair folders",
    )
    args = parser.parse_args()

    report_data = {
        "global_summary": {"total_pairs_evaluated": 0},
        "malware_results": [],
    }

    if not os.path.exists(args.slices_dir):
        print(f"[!] Directory {args.slices_dir} not found.")
        return

    for adv_base_name in os.listdir(args.slices_dir):
        pair_dir = os.path.join(args.slices_dir, adv_base_name)
        if not os.path.isdir(pair_dir):
            continue

        print(f"\n[*] Evaluating Adversarial Sample: {adv_base_name}")
        slice_files = os.listdir(pair_dir)
        num_pairs = len([f for f in slice_files if f.startswith("slice_orig_")])

        if num_pairs == 0:
            continue

        total_score = 0.0
        details = []
        matched_slices = 0

        print(
            f" [+] Spinning up ThreadPoolExecutor with {MAX_CONCURRENT_SLICES} workers..."
        )

        # Parallel Execution Block
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=MAX_CONCURRENT_SLICES
        ) as executor:
            future_to_idx = {}
            for i in range(num_pairs):
                orig_path = os.path.join(pair_dir, f"slice_orig_{i}.json")
                adv_path = os.path.join(pair_dir, f"slice_adv_{i}.json")
                fut = executor.submit(process_slice_task, i, orig_path, adv_path)
                future_to_idx[fut] = i

            for future in concurrent.futures.as_completed(future_to_idx):
                result = future.result()
                score = result["score"]

                total_score += score
                if score == 1.0:
                    matched_slices += 1

                print(
                    f"  -> Slice [{result['slice_index']}]: Score {score} ({result['reason']})"
                )
                details.append(result)

        # Sort details back into chronological order since threads complete randomly
        details = sorted(details, key=lambda x: x["slice_index"])

        final_similarity = (total_score / num_pairs) * 100

        report_data["malware_results"].append(
            {
                "original_name": "extracted_from_subset",
                "adversarial_name": adv_base_name,
                "total_sliced": num_pairs,
                "match": matched_slices,
                "similarity_score_percentage": round(final_similarity, 2),
                "slice_details": details,
            }
        )
        report_data["global_summary"]["total_pairs_evaluated"] += 1
        print(f"[+] {adv_base_name} Final Similarity: {final_similarity:.2f}%")

    with open(REPORT_FILENAME, "w") as f:
        json.dump(report_data, f, indent=4)
    print(f"\n[*] Complete! JSON Report saved to {REPORT_FILENAME}")


if __name__ == "__main__":
    main()
