import os
import numpy as np
import gc
import warnings
from tqdm import tqdm

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Suppress all the NumPy 2.x/3.14 unpickling spam
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
FUSION_DIR = "Fusion_table"
OUTPUT_DIR = "extracted"

# Ensure mapping is correct
# DATASETS_CONFIG = {
#     "gamma_adv": {"static": "gamma_static.npz", "dynamic": "gamma_dynamic.npz"},
#     "gapgan": {"static": "gapgan_static.npz", "dynamic": "gapgan_dynamic.npz"},
#     "MAB-mal": {"static": "mab_static.npz", "dynamic": "mab_dynamic.npz"},
#     "Malgpt": {"static": "malgpt_static.npz", "dynamic": "malgpt_dynamic.npz"},
#     "OBFU-mal": {"static": "obfu_static.npz", "dynamic": "obfu_dynamic.npz"},
#     "dqef": {"static": "dqef_static.npz", "dynamic": "dqef_dynamic.npz"},
#     "Test":{"static":"test_static.npz", "dynamic":"test_dynamic.npz"},
#     "Target":{"static":"dynamic_static.npz", "dynamic":"target_dynamic.npz"}
# }
#
# DATASETS_CONFIG = {
#     "aimed":{"static":"aimed_static.npz","dynamic":"aimed_dynamic.npz"},
#     "proposal":{"static":"proposal_static.npz","dynamic":"proposal_dynamic.npz"},
#     "MAB":{"static":"MAB_static.npz", "dynamic":"MAB_dynamic.npz"},
#     "gamma_adv": {"static": "gamma_static.npz", "dynamic": "gamma_dynamic.npz"},
#     "gapgan": {"static": "gapgan_static.npz", "dynamic": "gapgan_dynamic.npz"},
#     "Malgpt": {"static": "malgpt_static.npz", "dynamic": "malgpt_dynamic.npz"},
#     "OBFU-mal": {"static": "obfu_static.npz", "dynamic": "obfu_dynamic.npz"},
#     "dqef": {"static": "dqef_static.npz", "dynamic": "dqef_dynamic.npz"},
#     "Test":{"static":"test_static.npz", "dynamic":"test_dynamic.npz"}
# }

#
DATASETS_CONFIG = {

    "malguise": {"static": "malguise_static.npz", "dynamic": "malguise_dynamic.npz"}
}


def clean_name(name):
    if isinstance(name, (bytes, np.bytes_)):
        name = name.decode("utf-8", errors="ignore").strip()
    name = os.path.basename(str(name).strip())
    return os.path.splitext(name)[0]


def load_npz_to_pool(filepath, mode):
    if not os.path.exists(filepath):
        print(f" [!] Missing: {filepath}")
        return {}

    pool = {}
    try:
        # Check memory
        fast_mode = True
        if HAS_PSUTIL:
            free_gb = psutil.virtual_memory().available / (1024**3)
            file_size_gb = os.path.getsize(filepath) / (1024**3)
            # If free memory is less than ~3x the file size + 2GB buffer, fall back to safe mode
            if free_gb < (file_size_gb * 3) + 2:
                fast_mode = False
                print(
                    f" [*] Low memory ({free_gb:.1f} GB free). Using SAFE mode for {filepath}."
                )
            else:
                print(
                    f" [*] Memory sufficient ({free_gb:.1f} GB free). Using FAST mode for {filepath}."
                )

        data = np.load(filepath, allow_pickle=True)
        names = data["name"]

        if fast_mode:
            # We load the whole file object properties to RAM
            if mode == "static":
                raw_bytes = (
                    data["raw_byte"]
                    if "raw_byte" in data
                    else [np.array([], dtype=np.uint8)] * len(names)
                )
                opcodes = data["op_code"] if "op_code" in data else [""] * len(names)
                api_key = (
                    "api"
                    if "api" in data
                    else ("api_pefile" if "api_pefile" in data else None)
                )
                api_pe = data[api_key] if api_key else [""] * len(names)

                for i in range(len(names)):
                    h = clean_name(names[i])
                    pool[h] = {
                        "rawbyte": raw_bytes[i],
                        "opcode": opcodes[i],
                        "api_pefile": api_pe[i],
                    }

            else:  # mode == "dynamic"
                pe_imp = (
                    data["pe_imports"] if "pe_imports" in data else [[]] * len(names)
                )
                pe_sec = (
                    data["pe_sections"] if "pe_sections" in data else [[]] * len(names)
                )
                api_cu = data["api"] if "api" in data else [[]] * len(names)

                for i in range(len(names)):
                    h = clean_name(names[i])
                    pool[h] = {
                        "pe_imports": pe_imp[i],
                        "pe_sections": pe_sec[i],
                        "api_cuckoo": api_cu[i],
                    }
            pool["_fast_mode"] = True
            # We can release the NpzFile handle
            data.close()

        else:
            # SAFE MODE: keep mappings, keep data open
            mapping = {clean_name(names[i]): i for i in range(len(names))}
            pool["_fast_mode"] = False
            pool["_mapping"] = mapping
            pool["_data"] = data

    except Exception as e:
        print(f" [!] Error reading {filepath}: {e}")

    gc.collect()
    return pool


def fuse_dataset(ds_folder, config):
    print(f"\n{'=' * 60}\n PROCESSING: {ds_folder.upper()}\n{'=' * 60}")
    if not os.path.exists(ds_folder):
        print(f" [!] Physical directory '{ds_folder}' not found. Skipping.")
        return

    # 1. Load data into memory pools
    s_pool = load_npz_to_pool(os.path.join(FUSION_DIR, config["static"]), "static")
    d_pool = load_npz_to_pool(os.path.join(FUSION_DIR, config["dynamic"]), "dynamic")

    # 2. Preparation
    final = {
        "name": [],
        "label": [],
        "pe_imports": [],
        "pe_sections": [],
        "api_cuckoo": [],
        "api_pefile": [],
        "op_code": [],
        "raw_byte": [],
    }
    match_count, missing_s, missing_d = 0, 0, 0

    # 3. Cross-reference with physical files
    # Collect all file paths first
    tasks = []
    for label in os.listdir(ds_folder):
        label_path = os.path.join(ds_folder, label)
        if os.path.isdir(label_path):
            for fname in os.listdir(label_path):
                if not fname.startswith("."):
                    tasks.append((fname, label))

    print(f" [*] Fusing {len(tasks)} samples...")
    for fname, label in tqdm(tasks, desc="Merging"):
        h = clean_name(fname)
        final["name"].append(fname)
        final["label"].append(label)

        # Static
        if "_fast_mode" in s_pool and not s_pool["_fast_mode"]:
            _mapping, _data = s_pool.get("_mapping", {}), s_pool.get("_data")
            if h in _mapping and _data is not None:
                idx = _mapping[h]
                raw_bytes_arr = _data["raw_byte"] if "raw_byte" in _data else None
                opcodes_arr = _data["op_code"] if "op_code" in _data else None

                api_key = (
                    "api"
                    if "api" in _data
                    else ("api_pefile" if "api_pefile" in _data else None)
                )
                api_pe_arr = _data[api_key] if api_key else None

                final["raw_byte"].append(
                    raw_bytes_arr[idx]
                    if raw_bytes_arr is not None
                    else np.array([], dtype=np.uint8)
                )
                final["op_code"].append(
                    opcodes_arr[idx] if opcodes_arr is not None else ""
                )
                final["api_pefile"].append(
                    api_pe_arr[idx] if api_pe_arr is not None else ""
                )
            else:
                final["raw_byte"].append(np.array([], dtype=np.uint8))
                final["op_code"].append("")
                final["api_pefile"].append("")
                missing_s += 1
            has_s = h in _mapping
        else:
            if h in s_pool and h != "_fast_mode":
                final["raw_byte"].append(s_pool[h]["rawbyte"])
                final["op_code"].append(s_pool[h]["opcode"])
                final["api_pefile"].append(s_pool[h]["api_pefile"])
                has_s = True
            else:
                final["raw_byte"].append(np.array([], dtype=np.uint8))
                final["op_code"].append("")
                final["api_pefile"].append("")
                missing_s += 1
                has_s = False

        # Dynamic
        if "_fast_mode" in d_pool and not d_pool["_fast_mode"]:
            _mapping, _data = d_pool.get("_mapping", {}), d_pool.get("_data")
            if h in _mapping and _data is not None:
                idx = _mapping[h]
                pe_imports_arr = _data["pe_imports"] if "pe_imports" in _data else None
                pe_sections_arr = (
                    _data["pe_sections"] if "pe_sections" in _data else None
                )
                api_cu_arr = _data["api"] if "api" in _data else None

                final["pe_imports"].append(
                    pe_imports_arr[idx] if pe_imports_arr is not None else []
                )
                final["pe_sections"].append(
                    pe_sections_arr[idx] if pe_sections_arr is not None else []
                )
                final["api_cuckoo"].append(
                    api_cu_arr[idx] if api_cu_arr is not None else []
                )
            else:
                final["pe_imports"].append([])
                final["pe_sections"].append([])
                final["api_cuckoo"].append([])
                missing_d += 1
            has_d = h in _mapping
        else:
            if h in d_pool and h != "_fast_mode":
                final["pe_imports"].append(d_pool[h]["pe_imports"])
                final["pe_sections"].append(d_pool[h]["pe_sections"])
                final["api_cuckoo"].append(d_pool[h]["api_cuckoo"])
                has_d = True
            else:
                final["pe_imports"].append([])
                final["pe_sections"].append([])
                final["api_cuckoo"].append([])
                missing_d += 1
                has_d = False

        if has_s and has_d:
            match_count += 1

    # 4. Save
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    out_path = os.path.join(OUTPUT_DIR, f"{ds_folder}_full.npz")

    print(f" [*] Compressing and saving to {out_path}...")
    np.savez_compressed(
        out_path,
        name=np.array(final["name"]),
        label=np.array(final["label"]),
        pe_imports=np.array(final["pe_imports"], dtype=object),
        pe_sections=np.array(final["pe_sections"], dtype=object),
        api_cuckoo=np.array(final["api_cuckoo"], dtype=object),
        api_pefile=np.array(final["api_pefile"], dtype=object),
        op_code=np.array(final["op_code"], dtype=object),
        raw_byte=np.array(final["raw_byte"], dtype=object),
    )

    print(
        f" [DONE] Matches: {match_count} | Missing S: {missing_s} | Missing D: {missing_d}"
    )
    del s_pool, d_pool, final
    gc.collect()


def main():
    for ds_folder, config in DATASETS_CONFIG.items():
        fuse_dataset(ds_folder, config)


if __name__ == "__main__":
    main()
