import os
import glob
import json
import numpy as np
import re
import concurrent.futures
import time

try:
    from triton import TritonContext, ARCH, Instruction, AST_REPRESENTATION, MODE, MemoryAccess
    from z3 import Solver, unsat, sat, parse_smt2_string
    HAS_TRITON_Z3 = True
except ImportError:
    print(" [!] Missing triton or z3-solver. Ensure they are installed.")
    HAS_TRITON_Z3 = False

try:
    import fast_sw
    HAS_FAST_SW = True
except ImportError:
    print(" [!] Missing fast_sw module. Fallback Layer 3 will fail.")
    HAS_FAST_SW = False

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from examples.sequence_deduplicator import deduplicate_sequence

DATASETS_API_DIR = "datasets/APIs"
DATASETS_SYS_DIR = "datasets/syscalls"
THRESHOLD = 0.8
MAX_CONCURRENT_SLICES = 14
Z3_THREADS = 4
Z3_TIMEOUT_MS = 55000
CRYPTO_THRESHOLD_XOR = 200
CRYPTO_THRESHOLD_SHL = 200

SYSCALL_ARG_COUNTS = {
    0x42: 11, 0x30: 6, 0x0C: 1, 0x112: 9, 0xFE: 9, 0x13: 1,
    0x1D: 7, 0x0F: 3, 0x5D: 6, 0x3F: 2, 0x15: 6, 0x37: 5,
    0x10A: 10, 0x4D: 5, 0x12A: 2, 0x2F: 8, 0x23: 4, 0x35: 8,
    0xCE: 2, 0x114: 5, 0xD5: 2, 0x07: 10,
}

CRITICAL_APIS = {
    "NtCreateProcess", "NtOpenProcess", "NtTerminateProcess",
    "NtCreateThread", "NtResumeThread", "NtTerminateThread",
    "NtCreateFile", "NtOpenFile", "NtClose", 
    "NtQueryDirectoryFile", "NtSetInformationFile",
    "NtCreateKey", "NtOpenKey", "NtSaveKey",
    "NtAllocateVirtualMemory", "NtMapViewOfSection", "NtWriteVirtualMemory",
    "connect", "bind", "send", "recv", "gethostname",
    "CreateDesktop", "SwitchDesktop", "SetThreadDesktop",
    "LoadLibrary", "GetProcAddress", "GetModuleHandle"
}

def get_base_name(filename):
    if isinstance(filename, bytes):
        filename = filename.decode('utf-8', errors='ignore')
    elif hasattr(filename, 'item'):
        # In case it's a numpy bytes scalar
        val = filename.item()
        if isinstance(val, bytes):
            filename = val.decode('utf-8', errors='ignore')
        else:
            filename = str(val)
    else:
        filename = str(filename)
        
    match = re.search(r'([a-fA-F0-9]{32,64})', filename)
    if match:
        return match.group(1)
    return filename.split('.exe')[0] + '.exe'

def get_features(data, key_choices, count):
    for key in key_choices:
        if key in data.files:
            val = data[key]
            return val.tolist() if hasattr(val, 'tolist') else list(val)
    return [[] for _ in range(count)]

class FunctionalityEvaluator:
    def __init__(self):
        pass

    def check_integrity(self, api_chain, trace_count):
        api_ok = len(api_chain) > 0
        sys_ok = trace_count > 2
        is_alive = api_ok or sys_ok
        return is_alive, sys_ok

    def python_smith_waterman_align(self, seq1, seq2):
        n = len(seq1)
        m = len(seq2)
        if n == 0 or m == 0: return []
        
        CRITICAL_SYSCALLS = set(SYSCALL_ARG_COUNTS.keys())
        def get_score(a, b):
            if a["id"] != b["id"]: return -2.0
            weight = 2.0 if a["id"] in CRITICAL_SYSCALLS else 1.0
            return 3.0 * weight

        H = [[0.0] * (m + 1) for _ in range(n + 1)]
        tb = [[0] * (m + 1) for _ in range(n + 1)]
        
        max_score = 0.0
        max_i, max_j = 0, 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                match = H[i-1][j-1] + get_score(seq1[i-1], seq2[j-1])
                delete = H[i-1][j] - 2.0
                insert = H[i][j-1] - 2.0
                
                H[i][j] = max(0.0, match, delete, insert)
                
                if H[i][j] == 0: tb[i][j] = 0
                elif H[i][j] == match: tb[i][j] = 1
                elif H[i][j] == delete: tb[i][j] = 2
                else: tb[i][j] = 3
                
                if H[i][j] > max_score:
                    max_score = H[i][j]
                    max_i, max_j = i, j

        aligned_pairs = []
        i, j = max_i, max_j
        while i > 0 and j > 0 and H[i][j] > 0:
            if tb[i][j] == 1:
                aligned_pairs.append((i-1, j-1))
                i -= 1; j -= 1
            elif tb[i][j] == 2: i -= 1
            elif tb[i][j] == 3: j -= 1
            else: break
            
        aligned_pairs.reverse()
        return aligned_pairs

    def is_fake_dependency(self, entry: dict) -> bool:
        args = entry.get("args", [])
        if not args:
            return False
        try:
            int_args = [int(a, 16) if isinstance(a, str) else int(a) for a in args]
            return all(a <= 1 for a in int_args)
        except Exception:
            return False

    def filter_trace(self, raw_trace):
        filtered = []
        CRITICAL_SYSCALLS = set(SYSCALL_ARG_COUNTS.keys())
        for global_idx, entry in enumerate(raw_trace):
            entry["global_index"] = global_idx
            syscall_id = entry.get("id")
            if syscall_id not in CRITICAL_SYSCALLS:
                continue
            if self.is_fake_dependency(entry):
                continue
            filtered.append(entry)
        return filtered

    def extract_slice(self, raw_trace, global_idx):
        if global_idx >= len(raw_trace): return None
        target = raw_trace[global_idx]
        prior_returns = []
        for dep in target.get("deps", []):
            src_idx = dep.get("from_syscall_idx")
            if src_idx is not None and src_idx < len(raw_trace):
                src_call = raw_trace[src_idx]
                if src_call.get("ret"):
                    prior_returns.append({"from_syscall_idx": src_idx, "retval": src_call["ret"]})

        return {
            "target_syscall": {"id": target["id"], "pc": target["pc"]},
            "syscall_args_concrete": target.get("args", []),
            "prior_syscall_returns": prior_returns,
            "instruction_segment": target.get("instruction_segment", []),
            "memory_log": target.get("memory_log", []),
        }

    def lift_to_smt_from_dict(self, slice_data):
        instructions = slice_data.get("instruction_segment", [])
        args_concrete = slice_data.get("syscall_args_concrete", [])
        memory_log = slice_data.get("memory_log", [])
        target_id = slice_data.get("target_syscall", {}).get("id", 0)

        if not instructions:
            if not args_concrete: return "(_ bv0 32)"
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
        initial_edx_val = int(first_regs.get("edx", "0x0"), 16) if "edx" in first_regs else 0

        prior_retvals = set()
        for ret in slice_data.get("prior_syscall_returns", []):
            val = int(ret.get("retval", "0x0"), 16)
            if val != 0: prior_retvals.add(val)

        for r_name in ["eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi"]:
            reg_obj = getattr(ctx.registers, r_name)
            val = int(first_regs.get(r_name, "0x0"), 16)
            ctx.setConcreteRegisterValue(reg_obj, val)
            if val in prior_retvals and val != 0:
                ctx.symbolizeRegister(reg_obj, f"REG_{r_name.upper()}_TAINT")
            elif r_name in ["eax", "ebx", "ecx", "edx"]:
                ctx.symbolizeRegister(reg_obj, f"REG_{r_name.upper()}")

        for mem_entry in memory_log:
            try:
                addr, size, value = int(mem_entry["addr"], 16), int(mem_entry["size"]), int(mem_entry["value"], 16)
                ctx.setConcreteMemoryValue(MemoryAccess(addr, size), value)
                if mem_entry.get("type") == "R" and value in prior_retvals and value != 0:
                    ctx.symbolizeMemory(MemoryAccess(addr, size), f"MEM_{hex(addr)}")
            except: continue

        for block in instructions:
            try:
                pc, raw_bytes = int(block["pc"], 16), bytes.fromhex(block["bytes"])
                block_size = block.get("size", len(raw_bytes))
                offset = 0
                while offset < min(block_size, len(raw_bytes)):
                    chunk = raw_bytes[offset : offset + 15]
                    if not chunk: break
                    inst = Instruction(pc + offset, chunk)
                    ctx.processing(inst)
                    disas = inst.getDisassembly().lower()
                    offset += inst.getSize() if inst.getSize() > 0 else 1
                    if "sysenter" in disas or "int 0x2e" in disas: break
            except: continue

        arg_count = SYSCALL_ARG_COUNTS.get(target_id, 4)
        arg_formulas = []
        for j in range(arg_count):
            addr = initial_edx_val + 4 + (j * 4)
            try:
                arg_formulas.append(str(ctx.getMemoryAst(MemoryAccess(addr, 4))))
            except:
                arg_formulas.append("(_ bv0 32)")

        if not arg_formulas: return "(_ bv0 32)"
        combined = arg_formulas[-1]
        for f in reversed(arg_formulas[:-1]):
            combined = f"(concat {f} {combined})"
        formula_str = combined

        if formula_str.count("bvxor") > CRYPTO_THRESHOLD_XOR or formula_str.count("bvshl") > CRYPTO_THRESHOLD_SHL:
            return "[CRYPTO_DETECTED]"

        unique_slots = list(dict.fromkeys(re.compile(r"MEM_0x[0-9a-f]+").findall(formula_str)))
        for i, slot in enumerate(unique_slots):
            formula_str = formula_str.replace(slot, f"NORM_SLOT_{i}")

        unique_refs = list(dict.fromkeys(re.compile(r"ref!\d+").findall(formula_str)))
        for i, ref in enumerate(unique_refs):
            formula_str = formula_str.replace(ref, f"REF_{i}")

        return formula_str

    def load_expr(self, formula_str):
        known_vars = set(re.findall(r"(REG_[A-Z0-9_]+|SymVar_[0-9]+|NORM_SLOT_[0-9]+|ref![0-9]+|k![0-9]+|INIT_MEM_[a-f0-9x]+|FALLBACK_ARG_[0-9]+)", formula_str))
        decls_map = {v: "(_ BitVec 32)" for v in known_vars}
        current_probe_size = "32"

        while True:
            decls = "".join(f"(declare-const {v} {sort})\n" for v, sort in sorted(decls_map.items()))
            smt2_script = f"(set-logic QF_BV)\n{decls}\n(declare-const __probe__ (_ BitVec {current_probe_size}))\n(assert (= __probe__ {formula_str}))\n(check-sat)"
            try:
                assertions = parse_smt2_string(smt2_script)
                if not assertions: return None
                return assertions[0].arg(1)
            except Exception as e:
                error_msg = str(e)
                if "are incompatible" in error_msg:
                    sizes = re.findall(r"\(_ BitVec (\d+)\)", error_msg)
                    if len(sizes) >= 2:
                        target_size = sizes[0] if sizes[0] != current_probe_size else sizes[1]
                        if target_size == current_probe_size: return None
                        current_probe_size = target_size
                        continue
                match_const = re.search(r"unknown constant (ref!\d+|k!\d+|[a-zA-Z0-9_!]+)", error_msg)
                if match_const:
                    decls_map[match_const.group(1)] = "(_ BitVec 32)"
                    continue
                return None

    def evaluate_pair(self, orig_raw, adv_raw):
        if not orig_raw or not adv_raw: return 0.5
        if "[CRYPTO_DETECTED]" in orig_raw or "[CRYPTO_DETECTED]" in adv_raw: return 0.7
        if orig_raw == adv_raw: return 1.0

        expr_orig = self.load_expr(orig_raw)
        expr_adv = self.load_expr(adv_raw)
        if expr_orig is None or expr_adv is None or expr_orig.sort() != expr_adv.sort():
            return 0.5

        s1 = Solver()
        s1.set("timeout", Z3_TIMEOUT_MS)
        s1.set("threads", Z3_THREADS)
        s1.add(expr_orig != expr_adv)
        try:
            if s1.check() == unsat: return 1.0
            s2 = Solver()
            s2.set("timeout", Z3_TIMEOUT_MS)
            s2.set("threads", Z3_THREADS)
            s2.add(expr_orig == expr_adv)
            if s2.check() == sat: return 0.5
            return 0.0
        except:
            return 0.5

    def process_slice_task(self, slice_orig, slice_adv):
        try:
            formula_orig = self.lift_to_smt_from_dict(slice_orig)
            formula_adv = self.lift_to_smt_from_dict(slice_adv)
            return self.evaluate_pair(formula_orig, formula_adv)
        except Exception as e:
            return 0.5

    def run_binsim_layer(self, raw_ori, raw_adv):
        filt_ori = self.filter_trace(raw_ori)
        filt_adv = self.filter_trace(raw_adv)
        
        if not filt_ori or not filt_adv:
            raise Exception("Traces empty after filtering")

        aligned = self.python_smith_waterman_align(filt_ori, filt_adv)
        if not aligned:
            raise Exception("No semantic alignment islands found")

        total_score = 0.0
        scores = []
        for g_idx_ori, g_idx_adv in aligned:
            slice_orig = self.extract_slice(raw_ori, filt_ori[g_idx_ori]["global_index"])
            slice_adv = self.extract_slice(raw_adv, filt_adv[g_idx_adv]["global_index"])
            if slice_orig and slice_adv:
                scores.append(self.process_slice_task(slice_orig, slice_adv))

        if not scores:
            raise Exception("No valid slices extracted")

        for s in scores:
            total_score += s

        return (total_score / len(scores)) * 100.0

    def convert_to_apicall_objects(self, raw_sequence):
        api_objects = []
        for item in raw_sequence:
            name, attributes = "", ""
            if isinstance(item, dict):
                name = item.get("name", item.get("api", ""))
                args = item.get("arguments", {})
                if isinstance(args, dict): attributes = "|".join([f"{k}:{v}" for k, v in args.items()])
                elif isinstance(args, list): attributes = "|".join([str(x) for x in args])
            elif isinstance(item, str):
                name = item
                
            if not name: continue
            api_objects.append(fast_sw.ApiCall(name, attributes, name in CRITICAL_APIS))
        return api_objects

    def run_api_layer(self, ori_api, adv_api):
        if not HAS_FAST_SW:
            raise Exception("fast_sw is not installed!")
        clean_ori = deduplicate_sequence(ori_api, lm=5, k=2)
        clean_adv = deduplicate_sequence(adv_api, lm=5, k=2)
        
        c_ori = self.convert_to_apicall_objects(clean_ori)
        c_adv = self.convert_to_apicall_objects(clean_adv)
        
        if not c_ori or not c_adv: return 0.0
        return fast_sw.calculate_similarity(c_ori, c_adv)

def extract_json_if_needed(data):
    if hasattr(data, 'tolist'):
        data = data.tolist()
    if isinstance(data, str):
        try:
            return json.loads(data)
        except:
            return []
    return data if data is not None else []

def main():
    print(f"[*] Loading baseline datasets from {DATASETS_SYS_DIR} and {DATASETS_API_DIR}...")
    
    # Load original syscall baseline
    orig_sys_path = os.path.join(DATASETS_SYS_DIR, "Test_syscall.npz")
    orig_api_path = os.path.join(DATASETS_API_DIR, "Test_full.npz")
    
    if not os.path.exists(orig_sys_path) or not os.path.exists(orig_api_path):
        print("[!] Original baseline datasets not found.")
        return
        
    orig_sys_data = np.load(orig_sys_path, allow_pickle=True)
    orig_api_data = np.load(orig_api_path, allow_pickle=True)
    
    orig_sys_names = get_features(orig_sys_data, ['name'], 0)
    orig_sys_traces = get_features(orig_sys_data, ['trace_data'], len(orig_sys_names))
    orig_sys_counts = get_features(orig_sys_data, ['trace_count'], len(orig_sys_names))
    orig_sys_labels = get_features(orig_sys_data, ['label'], len(orig_sys_names))
    
    orig_api_names = get_features(orig_api_data, ['name'], 0)
    orig_api_chains = get_features(orig_api_data, ['api_cuckoo', 'api'], len(orig_api_names))
    orig_api_labels = get_features(orig_api_data, ['label'], len(orig_api_names))
    
    baseline = {}
    for name, trace, tcount, label in zip(orig_sys_names, orig_sys_traces, orig_sys_counts, orig_sys_labels):
        base = get_base_name(name)
        lbl_str = label.decode('utf-8') if isinstance(label, bytes) else (str(label) if label != [] else "")
        baseline[base] = {
            "trace_data": extract_json_if_needed(trace),
            "trace_count": int(tcount) if tcount is not None else 0,
            "api_chain": [],
            "label": lbl_str
        }
        
    for name, api, label in zip(orig_api_names, orig_api_chains, orig_api_labels):
        base = get_base_name(name)
        lbl_str = label.decode('utf-8') if isinstance(label, bytes) else (str(label) if label != [] else "")
        if base in baseline:
            baseline[base]["api_chain"] = extract_json_if_needed(api)
            if not baseline[base]["label"]:
                baseline[base]["label"] = lbl_str
        else:
            baseline[base] = {
                "trace_data": [],
                "trace_count": 0,
                "api_chain": extract_json_if_needed(api),
                "label": lbl_str
            }
            
    print(f"    Loaded {len(baseline)} original baseline samples.")

    # Find adversarial datasets
    evaluator = FunctionalityEvaluator()
    detailed_reports = {}
    summaries = {}
    
    # Dynamically find all API datasets
    api_files = glob.glob(os.path.join(DATASETS_API_DIR, "*.npz"))
    
    for api_file in api_files:
        api_filename = os.path.basename(api_file)
        if api_filename in ["Test_full.npz", "Target_full.npz"]:
            continue
            
        # Extract prefix (e.g. 'gamma' from 'gamma_full.npz')
        prefix = api_filename.replace("_full.npz", "").replace(".npz", "")
        
        # Try to find the matching syscall file, handling typos like 'sycall' and case mismatches
        syscall_pattern = re.compile(f"^{re.escape(prefix)}_sy.*\\.npz$", re.IGNORECASE)
        matching_syscalls = [f for f in os.listdir(DATASETS_SYS_DIR) if syscall_pattern.match(f)]
        
        dataset_name = prefix

        print(f"\n[*] Processing adversarial dataset: {dataset_name}")
        
        adv_sys_names = []
        adv_sys_traces = []
        adv_sys_counts = []

        if not matching_syscalls:
            print(f"    [!] Warning: Could not find matching syscall file for {api_filename}. Relying solely on API data.")
        else:
            sys_file = os.path.join(DATASETS_SYS_DIR, matching_syscalls[0])
            adv_sys_data = np.load(sys_file, allow_pickle=True)
            adv_sys_names = get_features(adv_sys_data, ['name'], 0)
            adv_sys_traces = get_features(adv_sys_data, ['trace_data'], len(adv_sys_names))
            adv_sys_counts = get_features(adv_sys_data, ['trace_count'], len(adv_sys_names))
        
        adv_api_data = np.load(api_file, allow_pickle=True)
        adv_api_names = get_features(adv_api_data, ['name'], 0)
        adv_api_chains = get_features(adv_api_data, ['api_cuckoo', 'api'], len(adv_api_names))
        
        adv_data_map = {}
        for name, trace, tcount in zip(adv_sys_names, adv_sys_traces, adv_sys_counts):
            base = get_base_name(name)
            adv_data_map[base] = {
                "name": name,
                "trace_data": extract_json_if_needed(trace),
                "trace_count": int(tcount) if tcount is not None else 0,
                "api_chain": []
            }
        
        for name, api in zip(adv_api_names, adv_api_chains):
            base = get_base_name(name)
            if base in adv_data_map:
                adv_data_map[base]["api_chain"] = extract_json_if_needed(api)
            else:
                adv_data_map[base] = {
                    "name": name,
                    "trace_data": [],
                    "trace_count": 0,
                    "api_chain": extract_json_if_needed(api)
                }

        dataset_report = {}
        integrity_count = 0
        functionality_count = 0
        total_samples = 0
        
        # Extended metrics for Q1 Paper
        layer2_total_score = 0.0
        layer3_total_score = 0.0
        layer2_eval_count = 0
        layer3_eval_count = 0
        layer2_functional_count = 0
        layer3_functional_count = 0
        
        for base, orig_info in baseline.items():
            if orig_info.get("label", "").lower() == "benign":
                continue
                
            total_samples += 1
            
            if base in adv_data_map:
                adv_info = adv_data_map[base]
            else:
                adv_info = {
                    "name": base,
                    "trace_data": [],
                    "trace_count": 0,
                    "api_chain": []
                }
            
            is_alive, adv_sys_ok = evaluator.check_integrity(adv_info["api_chain"], adv_info["trace_count"])
            orig_alive, orig_sys_ok = evaluator.check_integrity(orig_info["api_chain"], orig_info["trace_count"])
            
            if not is_alive or not orig_alive:
                dataset_report[str(adv_info["name"])] = {"score": 0.0, "method": "INTEGRITY_FAILED", "is_functional": False, "is_alive": False}
                continue
                
            integrity_count += 1 # Passed Layer 1 Integrity Gatekeeper
            score = 0.0
            method = ""
            force_api = not orig_sys_ok
            
            print(f"      -> Evaluating sample: {adv_info['name']} (Base: {base})", flush=True)
            if not force_api and adv_sys_ok and HAS_TRITON_Z3:
                try:
                    score = evaluator.run_binsim_layer(orig_info["trace_data"], adv_info["trace_data"])
                    method = "BINSIM_Z3"
                    layer2_eval_count += 1
                    layer2_total_score += score
                    print(f"         Layer 2 Score: {score:.2f}", flush=True)
                except Exception as e:
                    force_api = True
            else:
                force_api = True
                
            if force_api:
                if HAS_FAST_SW:
                    try:
                        print("         Triggering Layer 3...", flush=True)
                        score = evaluator.run_api_layer(orig_info["api_chain"], adv_info["api_chain"])
                        method = "API_SMITH_WATERMAN"
                        layer3_eval_count += 1
                        layer3_total_score += score
                        print(f"         Layer 3 Score: {score:.2f}", flush=True)
                    except Exception as e:
                        print(f" [!] Error in Layer 3: {e}", flush=True)
                else:
                    method = "FALLBACK_FAILED_NO_MODULE"
                    score = 0.0
                    
            is_functional = score >= (THRESHOLD * 100.0)
            
            if is_functional:
                functionality_count += 1
                if method == "BINSIM_Z3":
                    layer2_functional_count += 1
                elif method == "API_SMITH_WATERMAN":
                    layer3_functional_count += 1
                
            dataset_report[str(adv_info["name"])] = {
                "score": score,
                "method": method,
                "is_functional": is_functional,
                "is_alive": True
            }
            
        integrity_percentage = (integrity_count / total_samples * 100) if total_samples > 0 else 0.0
        functionality_percentage = (functionality_count / integrity_count * 100) if integrity_count > 0 else 0.0
        overall_percentage = (functionality_count / total_samples * 100) if total_samples > 0 else 0.0
        
        # Averages for research paper
        avg_layer2_score = (layer2_total_score / layer2_eval_count) if layer2_eval_count > 0 else 0.0
        avg_layer3_score = (layer3_total_score / layer3_eval_count) if layer3_eval_count > 0 else 0.0

        detailed_reports[dataset_name] = dataset_report
        summaries[dataset_name] = {
            "total_samples": total_samples,
            "integrity_count": integrity_count,
            "integrity_percentage": integrity_percentage,
            "functionality_count": functionality_count,
            "functionality_percentage": functionality_percentage,
            "overall_functionality_percentage": overall_percentage,
            "research_paper_stats": {
                "layer2_binsim_eval_count": layer2_eval_count,
                "layer2_functional_success": layer2_functional_count,
                "layer2_average_score": avg_layer2_score,
                "layer3_api_eval_count": layer3_eval_count,
                "layer3_functional_success": layer3_functional_count,
                "layer3_average_score": avg_layer3_score,
            }
        }
        
        print(f"    Total Samples: {total_samples} | Integrity Kept: {integrity_count} ({integrity_percentage:.2f}%) | Functionality Kept: {functionality_count} ({overall_percentage:.2f}%)")
        print(f"    [Research] L2 (BinSim) Evaluated: {layer2_eval_count} (Avg Score: {avg_layer2_score:.2f}) | L3 (API) Evaluated: {layer3_eval_count} (Avg Score: {avg_layer3_score:.2f})")

    print("\n[*] Saving comprehensive reports to comprehensive_detailed_reports.json")
    with open("comprehensive_detailed_reports.json", "w") as f:
        json.dump(detailed_reports, f, indent=4)
        
    print("[*] Saving summaries to comprehensive_summary.json")
    with open("comprehensive_summary.json", "w") as f:
        json.dump(summaries, f, indent=4)
        
    print("\n[+] Done!")

if __name__ == "__main__":
    main()
