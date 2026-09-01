import os
import json
import pyclamd
from pathlib import Path

# --- Configuration ---
DATASETS_DIR = r"C:\Users\Admin\Desktop\nghien_cuu_fiam\datasets"
# Save results locally in the same directory as the script
CURRENT_DIR = Path(__file__).parent.absolute()
PROGRESS_FILE = str(CURRENT_DIR / "clamav_results.json")
CHECKPOINT_INTERVAL = 10
PE_EXTENSIONS = {".exe", ".dll", ".sys", ".scr"}

def is_pe_file(file_path: Path) -> bool:
    """Verifies if a file is a Portable Executable via extension or MZ header."""
    if file_path.suffix.lower() in PE_EXTENSIONS:
        return True
    try:
        with open(file_path, "rb") as f:
            return f.read(2) == b"MZ"
    except Exception:
        return False

def load_progress() -> dict:
    """Loads existing progress JSON if available to resume scans."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[!] Warning: Could not read {PROGRESS_FILE} ({e}). Starting fresh.")
    return {"datasets": {}, "summary": {}}

def save_progress(data: dict):
    """Computes summary metrics and atomically saves progress to disk."""
    summary = {}
    for ds_name, ds_data in data.get("datasets", {}).items():
        results = ds_data.get("files", {})
        total = len(results)
        if total == 0:
            continue
        
        bypassed = sum(1 for item in results.values() if item["status"] == "bypassed")
        detected = sum(1 for item in results.values() if item["status"] == "detected")
        evasion_rate = (bypassed / total) * 100 if total > 0 else 0.0
        
        summary[ds_name] = {
            "total_pe_files": total,
            "detected_malware": detected,
            "bypassed_malware": bypassed,
            "evasion_rate_percent": round(evasion_rate, 2)
        }
    
    data["summary"] = summary
    
    # Atomic write to prevent file corruption during power loss
    temp_file = PROGRESS_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    os.replace(temp_file, PROGRESS_FILE)

def run_analysis():
    print("[*] Connecting to ClamAV Daemon...")
    try:
        cd = pyclamd.ClamdNetworkSocket(host="127.0.0.1", port=3310)
        if not cd.ping():
            print("[-] Error: ClamAV daemon is non-responsive. Check service status via 'net start clamd'.")
            return
    except Exception as e:
        print(f"[-] Connection Error: {e}")
        return

    print(f"[+] Connected to ClamAV Engine version: {cd.version().split()[0]}")

    data = load_progress()
    datasets_root = Path(DATASETS_DIR)

    if not datasets_root.exists():
        print(f"[-] Path does not exist: {DATASETS_DIR}")
        return

    # Retrieve dataset subdirectories (aimed, dqeaf, gamma, etc.)
    dataset_dirs = [d for d in datasets_root.iterdir() if d.is_dir()]
    print(f"[+] Located {len(dataset_dirs)} dataset folders inside '{DATASETS_DIR}'.\n")

    for ds_dir in dataset_dirs:
        ds_name = ds_dir.name
        print("=" * 60)
        print(f"[*] DATASET: {ds_name}")
        print("=" * 60)

        if ds_name not in data["datasets"]:
            data["datasets"][ds_name] = {"files": {}}

        scanned_files = data["datasets"][ds_name]["files"]
        
        # Discover all PE binaries recursively within the subfolder
        pe_files = [f for f in ds_dir.rglob("*") if f.is_file() and is_pe_file(f)]
        print(f"[+] Found {len(pe_files)} total PE files in '{ds_name}'.")

        unprocessed = [f for f in pe_files if str(f.absolute()) not in scanned_files]
        print(f"[*] Scanned previously: {len(pe_files) - len(unprocessed)} | Pending: {len(unprocessed)}")

        file_counter = 0

        for file_path in unprocessed:
            abs_path = str(file_path.absolute())
            try:
                # Scan file via socket connection
                scan_res = cd.scan_file(abs_path)
                
                if scan_res is None:
                    scanned_files[abs_path] = {
                        "filename": file_path.name,
                        "status": "bypassed",
                        "detection": None
                    }
                    print(f"  [+] BYPASSED : {file_path.name}")
                else:
                    virus_name = scan_res[abs_path][1]
                    scanned_files[abs_path] = {
                        "filename": file_path.name,
                        "status": "detected",
                        "detection": virus_name
                    }
                    print(f"  [-] DETECTED : {file_path.name} -> {virus_name}")

            except Exception as e:
                print(f"  [!] Exception scanning {file_path.name}: {e}")
                continue

            file_counter += 1

            # Save progress every 10 files
            if file_counter % CHECKPOINT_INTERVAL == 0:
                save_progress(data)
                print(f"  [✔ Checkpoint] Saved progress ({file_counter} new files processed)...")

        # Final save per dataset completion
        save_progress(data)

    # Output Final Summary Report
    print("\n" + "=" * 60)
    print("                 FINAL EVALUATION SUMMARY                  ")
    print("=" * 60)
    summary = data.get("summary", {})
    for ds_name, stats in summary.items():
        print(f"Dataset: {ds_name}")
        print(f"  ├── Total PE Files : {stats['total_pe_files']}")
        print(f"  ├── Detected       : {stats['detected_malware']}")
        print(f"  ├── Bypassed       : {stats['bypassed_malware']}")
        print(f"  └── Evasion Rate   : {stats['evasion_rate_percent']}%\n")

if __name__ == "__main__":
    run_analysis()