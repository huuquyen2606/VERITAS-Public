import os
import json
import subprocess
from pathlib import Path

# --- Configuration ---
DATASETS_DIR = r"C:\Users\Logan\Desktop\nghien_cuu_fiam\datasets"
# Save results locally in the same directory as the script
CURRENT_DIR = Path(__file__).parent.absolute()
PROGRESS_FILE = str(CURRENT_DIR / "defender_results.json")
CHECKPOINT_INTERVAL = 10
PE_EXTENSIONS = {".exe", ".dll", ".sys", ".scr"}
DEFENDER_PATH = r"C:\Program Files\Windows Defender\MpCmdRun.exe"

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
        errors = sum(1 for item in results.values() if item["status"] == "error")
        
        evasion_rate = (bypassed / total) * 100 if total > 0 else 0.0
        
        summary[ds_name] = {
            "total_pe_files": total,
            "detected_malware": detected,
            "bypassed_malware": bypassed,
            "scan_errors": errors,
            "evasion_rate_percent": round(evasion_rate, 2)
        }
    
    data["summary"] = summary
    
    # Atomic write to prevent file corruption during power loss
    temp_file = PROGRESS_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    os.replace(temp_file, PROGRESS_FILE)

def scan_with_defender(file_path: Path) -> tuple[str, str | None]:
    """
    Scans a single file using Windows Defender CLI.
    Return Codes:
      0 = No threat found
      2 = Threat identified
    """
    cmd = [
        DEFENDER_PATH,
        "-Scan",
        "-ScanType", "3",
        "-File", str(file_path.absolute()),
        "-DisableRemediation"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return "bypassed", None
        elif result.returncode == 2:
            # Parse detection details from output if available
            threat_info = "Threat Detected"
            for line in result.stdout.splitlines():
                if "THREAT" in line.upper() or "LISTED" in line.upper():
                    threat_info = line.strip()
                    break
            return "detected", threat_info
        else:
            return "error", f"MpCmdRun exit code: {result.returncode}"
    except Exception as e:
        return "error", str(e)

def run_analysis():
    print("[*] Verifying Windows Defender Binary...")
    if not os.path.exists(DEFENDER_PATH):
        print(f"[-] Error: Defender binary not found at path: {DEFENDER_PATH}")
        return
    print(f"[+] Located Defender CLI: {DEFENDER_PATH}")

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
            
            status, detection_info = scan_with_defender(file_path)
            
            scanned_files[abs_path] = {
                "filename": file_path.name,
                "status": status,
                "detection": detection_info
            }

            if status == "bypassed":
                print(f"  [+] BYPASSED : {file_path.name}")
            elif status == "detected":
                print(f"  [-] DETECTED : {file_path.name} -> {detection_info}")
            else:
                print(f"  [!] ERROR    : {file_path.name} -> {detection_info}")

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
        print(f"  ├── Errors         : {stats['scan_errors']}")
        print(f"  └── Evasion Rate   : {stats['evasion_rate_percent']}%\n")

if __name__ == "__main__":
    run_analysis()