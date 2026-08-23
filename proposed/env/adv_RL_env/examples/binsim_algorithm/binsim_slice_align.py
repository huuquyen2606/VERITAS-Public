import os
import json
import argparse
import subprocess
import numpy as np

# --- STRICT ACADEMIC SYSCALL SETTINGS ---
CRITICAL_SYSCALLS = {
    0x42,
    0x30,
    0x0C,
    0x112,
    0xFE,
    0x13,
    0x1D,
    0x0F,
    0x5D,
    0x3F,
    0x15,
    0x37,
    0x10A,
    0x4D,
    0x12A,
    0x2F,
    0x23,
    0x35,
    0xCE,
    0x114,
    0xD5,
    0x07,
}

SYSCALL_WEIGHT = {sc: 2 for sc in CRITICAL_SYSCALLS}
DEFAULT_WEIGHT = 1


def is_fake_dependency(entry: dict) -> bool:
    args = entry.get("args", [])
    if not args:
        return False
    try:
        int_args = [int(a, 16) for a in args]
        return all(a <= 1 for a in int_args)
    except Exception:
        return False


def filter_trace(raw_trace: list) -> list:
    filtered = []
    for global_idx, entry in enumerate(raw_trace):
        entry["global_index"] = global_idx
        syscall_id = entry.get("id")

        if syscall_id not in CRITICAL_SYSCALLS:
            continue
        if is_fake_dependency(entry):
            continue
        filtered.append(entry)
    return filtered


def write_alignment_files(path_prefix, trace_data):
    seq_path = f"{path_prefix}_seq.txt"
    wgt_path = f"{path_prefix}_wgt.txt"

    with open(seq_path, "w") as f:
        f.write(" ".join(str(x["id"]) for x in trace_data))

    with open(wgt_path, "w") as f:
        weights = [str(SYSCALL_WEIGHT.get(x["id"], DEFAULT_WEIGHT)) for x in trace_data]
        f.write(" ".join(weights))

    return seq_path, wgt_path


def extract_slice(raw_trace, global_idx):
    if global_idx >= len(raw_trace):
        return None
    target = raw_trace[global_idx]

    # Resolve prior return values using the original raw trace
    prior_returns = []
    for dep in target.get("deps", []):
        source_idx = dep.get("from_syscall_idx")
        if source_idx is not None and source_idx < len(raw_trace):
            source_call = raw_trace[source_idx]
            if source_call.get("ret"):
                prior_returns.append(
                    {"from_syscall_idx": source_idx, "retval": source_call["ret"]}
                )

    slice_data = {
        "target_syscall": {"id": target["id"], "pc": target["pc"]},
        "syscall_args_concrete": target.get("args", []),
        "prior_syscall_returns": prior_returns,
        "instruction_segment": target.get("instruction_segment", []),
        "memory_log": target.get("memory_log", []),
    }
    return slice_data


def process_pair(ori_name, adv_name, raw_ori, raw_adv, out_dir):
    print(f"\n[*] Processing Pair: {ori_name} -> {adv_name}")

    filt_ori = filter_trace(raw_ori)
    filt_adv = filter_trace(raw_adv)

    if not filt_ori or not filt_adv:
        print(f" [!] One or both traces empty after filtering. Skipping.")
        return 0

    # Setup temp files for C++ Aligner
    tmp_dir = os.path.join(out_dir, ".tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    seq_a, wgt_a = write_alignment_files(os.path.join(tmp_dir, "ori"), filt_ori)
    seq_b, wgt_b = write_alignment_files(os.path.join(tmp_dir, "adv"), filt_adv)

    cmd = ["./aligner", seq_a, seq_b, wgt_a, wgt_b]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Cleanup temp files
    for p in [seq_a, wgt_a, seq_b, wgt_b]:
        if os.path.exists(p):
            os.remove(p)

    if result.returncode != 0:
        print(f" [!] C++ Aligner failed for this pair.")
        return 0

    # Parse Smith-Waterman output
    matched_pairs = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        try:
            idx_a, idx_b = map(int, line.split(","))
            matched_pairs.append(
                (filt_ori[idx_a]["global_index"], filt_adv[idx_b]["global_index"])
            )
        except Exception:
            continue

    if not matched_pairs:
        print(" [!] No semantic alignment islands found.")
        return 0

    # Create dedicated output directory for this malware pair
    pair_dir = os.path.join(out_dir, os.path.splitext(adv_name)[0])
    os.makedirs(pair_dir, exist_ok=True)

    # Slice and Save
    success_slices = 0
    for pair_idx, (g_idx_ori, g_idx_adv) in enumerate(matched_pairs):
        slice_orig = extract_slice(raw_ori, g_idx_ori)
        slice_adv = extract_slice(raw_adv, g_idx_adv)

        if slice_orig and slice_adv:
            with open(os.path.join(pair_dir, f"slice_orig_{pair_idx}.json"), "w") as f:
                json.dump(slice_orig, f, indent=4)
            with open(os.path.join(pair_dir, f"slice_adv_{pair_idx}.json"), "w") as f:
                json.dump(slice_adv, f, indent=4)
            success_slices += 1

    print(f" [+] Generated {success_slices} synchronized slice pairs in '{pair_dir}/'")
    return success_slices


def load_npz_to_dict(npz_path):
    print(f"[*] Loading NPZ: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    names = data["name"]
    traces = data["trace_data"]

    dataset = {}
    for i in range(len(names)):
        # Only load if we have actual trace blocks
        if isinstance(traces[i], list) and len(traces[i]) > 0:
            dataset[names[i]] = traces[i]
    return dataset


def main():
    parser = argparse.ArgumentParser(description="BinSim NPZ Aligner & Slicer")
    parser.add_argument("--ori-npz", required=True, help="Original Dataset NPZ")
    parser.add_argument("--adv-npz", required=True, help="Adversarial Dataset NPZ")
    parser.add_argument(
        "--out-dir", default="binsim_slices", help="Output directory for Triton slices"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ori_data = load_npz_to_dict(args.ori_npz)
    adv_data = load_npz_to_dict(args.adv_npz)

    print(f"[*] Valid Original Samples : {len(ori_data)}")
    print(f"[*] Valid Adversarial Samples : {len(adv_data)}")

    total_matches = 0
    total_slices_generated = 0

    print("\n" + "=" * 55)
    print(">>> STAGE 2 & 3: BATCH ALIGNMENT & SLICING <<<")
    print("=" * 55)

    for adv_name, adv_trace in adv_data.items():
        adv_base = os.path.splitext(adv_name)[0]

        # String matching: Check if the original base name is a subset of the adversarial name
        matched_ori_name = None
        for ori_name in ori_data.keys():
            ori_base = os.path.splitext(ori_name)[0]
            if ori_base in adv_base:
                matched_ori_name = ori_name
                break

        if matched_ori_name:
            total_matches += 1
            ori_trace = ori_data[matched_ori_name]
            slices = process_pair(
                matched_ori_name, adv_name, ori_trace, adv_trace, args.out_dir
            )
            total_slices_generated += slices
        else:
            print(f" [!] Warning: No original match found for {adv_name}")

    print("\n" + "=" * 55)
    print(">>> SUMMARY <<<")
    print("=" * 55)
    print(f" Malware Pairs Matched : {total_matches}")
    print(f" Total Slices Ready for Triton: {total_slices_generated}")
    print(f" Output Directory : {args.out_dir}/")


if __name__ == "__main__":
    main()
