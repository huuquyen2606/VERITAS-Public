import os
import time
import requests
import numpy as np
import json
import glob
import sys
import io
import argparse
import pyzipper
import paramiko

# --- CAPEv2 CONFIGURATION ---
CAPE_HOST = "http://<YOUR_CAPE_NODE_IP>:8000"
API_TOKEN = "<YOUR_CAPE_API_TOKEN>"
HEADERS = {"Authorization": f"Token {API_TOKEN}"}

# --- SSH HARD CLEANUP CONFIGURATION ---
SSH_HOST = "<YOUR_CAPE_NODE_IP>"
SSH_USER = "cape"
SSH_PASS = "<YOUR_SSH_PASSWORD>"
CAPE_DIR = "/opt/CAPEv2"

WORKSPACE_DIR = "binsim_progress"
PROGRESS_LOG = os.path.join(WORKSPACE_DIR, "processed_log.json")
FINAL_OUTPUT_FILE = "binsim_final.npz"
BATCH_SIZE = 3


def ensure_workspace():
    if not os.path.exists(WORKSPACE_DIR):
        os.makedirs(WORKSPACE_DIR)


def load_processed_set():
    if os.path.exists(PROGRESS_LOG):
        try:
            with open(PROGRESS_LOG, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except:
            return set()
    return set()


def save_batch_npz(data_dict):
    if not data_dict["name"]:
        return
    count = len(data_dict["name"])
    timestamp = int(time.time())
    filename = f"batch_{timestamp}_{count}_samples.npz"
    filepath = os.path.join(WORKSPACE_DIR, filename)

    print(f"\n [SAVE] Packing {count} files into -> {filename}")

    try:
        np.savez_compressed(
            filepath,
            name=np.array(data_dict["name"]),
            label=np.array(data_dict["label"]),
            is_running=np.array(data_dict["is_running"]),
            trace_count=np.array(data_dict["trace_count"]),
            trace_data=np.array(data_dict["trace_data"], dtype=object),
        )
    except Exception as e:
        print(f" [!] NPZ SAVE ERROR: {e}")
        return

    current_log = []
    if os.path.exists(PROGRESS_LOG):
        try:
            with open(PROGRESS_LOG, "r", encoding="utf-8") as f:
                current_log = json.load(f)
        except:
            pass

    current_log.extend(data_dict["name"])
    try:
        with open(PROGRESS_LOG, "w", encoding="utf-8") as f:
            json.dump(current_log, f, indent=2)
    except Exception as e:
        pass


def merge_all_npz():
    print(f"\n--- FINAL STAGE: MERGING ALL BATCHES ---")
    npz_files = glob.glob(os.path.join(WORKSPACE_DIR, "*.npz"))
    if not npz_files:
        return

    all_data = {
        "name": [],
        "label": [],
        "is_running": [],
        "trace_count": [],
        "trace_data": [],
    }

    for f in npz_files:
        try:
            d = np.load(f, allow_pickle=True)
            for k in all_data.keys():
                all_data[k].extend(d[k])
        except Exception as e:
            print(f" [!] Error reading {f}: {e}")

    try:
        np.savez_compressed(
            FINAL_OUTPUT_FILE,
            name=np.array(all_data["name"]),
            label=np.array(all_data["label"]),
            is_running=np.array(all_data["is_running"]),
            trace_count=np.array(all_data["trace_count"]),
            trace_data=np.array(all_data["trace_data"], dtype=object),
        )
        print(f" [DONE] Merged dataset saved: {FINAL_OUTPUT_FILE}")
    except Exception as e:
        print(f" [!] Merge error: {e}")


def hard_clean_cape_server():
    print("\n [SSH] Initiating Hard Cleanup on CAPE Server...")
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(SSH_HOST, username=SSH_USER, password=SSH_PASS, timeout=10)

        commands = [
            'echo "db.dropDatabase()" | mongo cape || echo "db.dropDatabase()" | mongosh cape',
            f'echo "{SSH_PASS}" | sudo -S rm -rf {CAPE_DIR}/storage/analyses/*',
            f'echo "{SSH_PASS}" | sudo -S rm -rf {CAPE_DIR}/storage/binaries/*',
            f'echo "{SSH_PASS}" | sudo -S systemctl restart cape.service',
            f'echo "{SSH_PASS}" | sudo -S systemctl restart cape-processor.service',
            f'echo "{SSH_PASS}" | sudo -S systemctl restart cape-web.service',
        ]

        for cmd in commands:
            stdin, stdout, stderr = ssh.exec_command(cmd)
            stdout.channel.recv_exit_status()

        print(" [SSH] Cleanup Complete. Waiting 10s for CAPE to boot...")
        ssh.close()
        time.sleep(10)
    except Exception as e:
        print(f" [!] SSH Hard Clean Error: {e}")


def submit_to_cape(file_path):
    url = f"{CAPE_HOST}/apiv2/tasks/create/file/"
    options_str = "sniffer=0,procmemdump=0,dumpprocess=0,dump_r0=0,curtain=0,sysmon=0"
    data_params = {
        "timeout": 60,
        "platform": "windows",
        "machine": "win10",
        "package": "binsim",
        "enforce_timeout": False,
        "priority": 1,
        "options": options_str,
    }

    while True:
        try:
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f)}
                r = requests.post(
                    url, headers=HEADERS, files=files, data=data_params, timeout=30
                )
                if r.status_code == 200:
                    task_ids = r.json().get("data", {}).get("task_ids", [])
                    if task_ids:
                        return task_ids[0]
                elif r.status_code >= 500:
                    time.sleep(15)
                    continue
                else:
                    return None
        except Exception as e:
            time.sleep(10)


def wait_for_report(task_id):
    status_url = f"{CAPE_HOST}/apiv2/tasks/view/{task_id}/"
    report_url = f"{CAPE_HOST}/apiv2/tasks/get/report/{task_id}/json/"
    print(f"[*] Waiting on Task {task_id}...", end="", flush=True)

    for i in range(240):
        try:
            r = requests.get(status_url, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                status = r.json().get("data", {}).get("status")
                if status in ["reported", "failed_analysis", "timeout", "completed"]:
                    rep_req = requests.get(report_url, headers=HEADERS, timeout=60)
                    if rep_req.status_code == 200:
                        report_data = rep_req.json()
                        if report_data.get("error") is True:
                            time.sleep(5)
                            continue
                        print(" OK!")
                        return report_data
            time.sleep(10)
        except Exception:
            pass
    return {}


def delete_task(task_id):
    url = f"{CAPE_HOST}/apiv2/tasks/delete/{task_id}/"
    try:
        requests.get(url, headers=HEADERS, timeout=10)
    except:
        pass


def find_binsim_sha256(obj):
    if isinstance(obj, dict):
        name = obj.get("name") or obj.get("filepath") or obj.get("path")
        if name and "binsim_trace.json" in str(name):
            if "sha256" in obj:
                return obj["sha256"]
        for v in obj.values():
            result = find_binsim_sha256(v)
            if result:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = find_binsim_sha256(item)
            if result:
                return result
    return None


def extract_binsim_trace(report, task_id):
    assembly_slices = []
    sha256 = find_binsim_sha256(report)
    if not sha256:
        return assembly_slices

    dl_url = f"{CAPE_HOST}/apiv2/tasks/get/dropped/{task_id}/"
    try:
        r = requests.get(dl_url, headers=HEADERS, timeout=60)
        if r.status_code == 200:
            file_bytes = io.BytesIO(r.content)
            extracted_text = None
            try:
                with pyzipper.AESZipFile(file_bytes) as z:
                    z.setpassword(b"infected")
                    for name in z.namelist():
                        if name.endswith(".json") and (
                            sha256 in name or "binsim_trace" in name
                        ):
                            extracted_text = z.read(name).decode(
                                "utf-8", errors="ignore"
                            )
                            break
            except Exception as e:
                pass

            if extracted_text:
                for line in extracted_text.strip().split("\n"):
                    if line:
                        try:
                            assembly_slices.append(json.loads(line))
                        except Exception:
                            pass
    except Exception as e:
        pass
    return assembly_slices


def check_is_running(report, trace_count):
    if trace_count > 0:
        return True
    try:
        for proc in report.get("behavior", {}).get("processes", []):
            if len(proc.get("calls", [])) > 0:
                return True
    except:
        pass
    return False


def main():
    parser = argparse.ArgumentParser(
        description="BinSim NPZ Extractor with SSH Cleanup"
    )
    parser.add_argument("-f", "--file", help="Process a single file")
    parser.add_argument("-d", "--dir", help="Process a directory of files")
    args = parser.parse_args()

    if not args.file and not args.dir:
        print("[!] Provide either -f <file> or -d <dir>")
        sys.exit(1)

    ensure_workspace()
    processed_set = load_processed_set()

    files_to_process = []
    if args.file:
        files_to_process.append((args.file, "single"))
    elif args.dir:
        for root, _, files in os.walk(args.dir):
            for f in files:
                if not f.startswith("."):
                    files_to_process.append(
                        (os.path.join(root, f), os.path.basename(root))
                    )

    print(f"--- Loaded {len(processed_set)} previously processed files (Skipping) ---")

    current_batch = {
        k: [] for k in ["name", "label", "is_running", "trace_count", "trace_data"]
    }
    stats_total = 0
    stats_running = 0

    try:
        for file_path, label in files_to_process:
            filename = os.path.basename(file_path)
            if filename in processed_set:
                continue

            print(f"\n>>> File: {filename} | Label: {label}")
            start_time = time.time()

            tid = submit_to_cape(file_path)
            if not tid:
                continue

            report = wait_for_report(tid)
            frida_assembly = extract_binsim_trace(report, tid)
            trace_count = len(frida_assembly)

            # --- METRICS GATHERING ---
            is_running = check_is_running(report, trace_count)
            duration = time.time() - start_time

            stats_total += 1
            if is_running:
                stats_running += 1

            print(
                f"   [METRICS] Is_Running: {is_running} | Trace Blocks: {trace_count}"
            )
            print(f"   [TIME]    Finished in: {duration:.2f}s")

            # Store the raw JSON objects directly into the array (no disk writes for traces)
            current_batch["name"].append(filename)
            current_batch["label"].append(label)
            current_batch["is_running"].append(is_running)
            current_batch["trace_count"].append(trace_count)
            current_batch["trace_data"].append(frida_assembly)

            processed_set.add(filename)
            delete_task(tid)

            if len(current_batch["name"]) >= BATCH_SIZE:
                save_batch_npz(current_batch)
                hard_clean_cape_server()
                current_batch = {k: [] for k in current_batch}

    except KeyboardInterrupt:
        print("\n\n [STOP] HALTED BY USER...")
    finally:
        if len(current_batch["name"]) > 0:
            save_batch_npz(current_batch)
            hard_clean_cape_server()

        if args.dir:
            merge_all_npz()

        print("\n" + "=" * 60)
        print(">>> BINSIM EXTRACTION RUN SUMMARY <<<")
        print("=" * 60)
        print(f" Total Files Processed : {stats_total}")
        print(f" Files 'Running'       : {stats_running}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
