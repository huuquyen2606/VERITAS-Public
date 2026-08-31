# -*- coding: utf-8 -*-
"""
================================================================================
 ULTIMATE CAPEv2 EXTRACTION PIPELINE (AS_COMPLETED + FAST RESTART)
================================================================================
"""

import os
import time
import json
import glob
import codecs
import shutil
import io
from datetime import datetime
import concurrent.futures

import requests
import numpy as np
import paramiko
import pyzipper

# ==============================================================================
# 1. CLUSTER + CREDENTIALS
# ==============================================================================
# Because my cape setup does not clear the whole analysis results ( it build up over time)
# SO i need to have this autoclean up feature setting up for the script to manually erase past analysis data.
# TO prevent filling up spaces !

API_TOKEN = "<YourAPITokenHere"
SSH_USER = "username"
SSH_PASS = "passwd"
CAPE_DIR = "/opt/CAPEv2"

IP_LIST = [
    "192.168.2.122",
    "192.168.2.124",
    "192.168.2.116",
    "192.168.2.125",
    "192.168.2.129",
    "192.168.2.131",
]
# list of your CAPE servers, you can add here

SERVERS = [
    {
        "name": f"CAPE-{ip.split('.')[-1]}",
        "cape_host": f"http://{ip}:8000",
        "api_token": API_TOKEN,
        "ssh_host": ip,
        "ssh_user": SSH_USER,
        "ssh_pass": SSH_PASS,
        "cape_dir": CAPE_DIR,
    }
    for ip in IP_LIST
]

BATCH_SIZE = len(SERVERS)

# ==============================================================================
# 2. DATASETS
# ==============================================================================
DATASET_ROOT = "/home/cape/Downloads/new_proposal"
DATASETS = ["malguise"]

FAIL_PATH = "/home/cape/Downloads/new_proposal/corrupted_samples/"


def dataset_paths(dataset_name):
    return {
        "dir": os.path.join(DATASET_ROOT, dataset_name),
        "feat_workspace": f"{dataset_name}_features_progress",
        "feat_output": f"{dataset_name}_features.npz",
        "sys_workspace": f"{dataset_name}_syscall_progress",
        "sys_output": f"{dataset_name}_syscall.npz",
        "report_output": f"{dataset_name}_integrity_report.json",
    }


# ==============================================================================
# 3. AIMED-STYLE INTEGRITY CHECK
# ==============================================================================
AIMED_ERR_SIGNATURE = (
    "CuckooPackageError: Unable to execute the initial process, analysis aborted.\n"
)
AIMED_MIN_DURATION = 15


def is_sample_running(report, aux_count=0):
    if aux_count > 0:
        return True
    try:
        for proc in report.get("behavior", {}).get("processes", []):
            if len(proc.get("calls", [])) > 0:
                return True
    except Exception:
        pass
    return False


def check_integrity(report, duration):
    cuckoo_debug = report.get("debug", {}).get("cuckoo", "")
    if isinstance(cuckoo_debug, list):
        cuckoo_debug = "".join(cuckoo_debug)
    elif not isinstance(cuckoo_debug, str):
        cuckoo_debug = str(cuckoo_debug)

    report_duration = report.get("info", {}).get("duration", 0)
    if not report_duration:
        report_duration = duration

    if AIMED_ERR_SIGNATURE in cuckoo_debug or report_duration < AIMED_MIN_DURATION:
        return False
    return True


# ==============================================================================
# 4. WORKSPACE / PROGRESS / INTEGRITY STATE HELPERS
# ==============================================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_json_set(path):
    if os.path.exists(path):
        try:
            with codecs.open(path, "r", "utf-8") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()


def append_json_list(path, new_items):
    current = []
    if os.path.exists(path):
        try:
            with codecs.open(path, "r", "utf-8") as f:
                current = json.load(f)
        except Exception:
            pass
    current.extend(new_items)
    with codecs.open(path, "w", "utf-8") as f:
        json.dump(current, f, indent=2)


def load_integrity_state(workspace_dir):
    path = os.path.join(workspace_dir, "integrity_state.json")
    if os.path.exists(path):
        try:
            with codecs.open(path, "r", "utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_integrity_state(workspace_dir, updates):
    path = os.path.join(workspace_dir, "integrity_state.json")
    state = load_integrity_state(workspace_dir)
    state.update(updates)
    with codecs.open(path, "w", "utf-8") as f:
        json.dump(state, f, indent=2)


# ==============================================================================
# 5. NPZ BATCH SAVE / MERGE
# ==============================================================================
def save_batch_npz(data_dict, workspace_dir, object_fields=frozenset()):
    if not data_dict.get("name"):
        return
    count = len(data_dict["name"])
    timestamp = int(time.time())
    filename = f"batch_{timestamp}_{count}_samples.npz"
    filepath = os.path.join(workspace_dir, filename)

    print(f"\n [SAVE] Committing {count} samples -> {filepath}")
    try:
        arrays = {}
        for k, v in data_dict.items():
            arrays[k] = np.array(v, dtype=object) if k in object_fields else np.array(v)
        np.savez_compressed(filepath, **arrays)
    except Exception as e:
        print(f" [!] NPZ SAVE ERROR: {e}")
        return

    append_json_list(
        os.path.join(workspace_dir, "processed_log.json"), data_dict["name"]
    )


def merge_all_npz(workspace_dir, final_output_file, fields, object_fields=frozenset()):
    print(f"\n--- MERGING BATCHES -> {final_output_file} ---")
    npz_files = glob.glob(os.path.join(workspace_dir, "*.npz"))
    if not npz_files:
        print(f" [!] No .npz batches found in {workspace_dir}")
        return None

    all_data = {k: [] for k in fields}
    for f in npz_files:
        try:
            d = np.load(f, allow_pickle=True)
            for k in fields:
                items = d[k].tolist() if hasattr(d[k], "tolist") else list(d[k])
                all_data[k].extend(items)
        except Exception as e:
            print(f" [!] Error reading {f}: {e}")

    try:
        arrays = {
            k: (np.array(v, dtype=object) if k in object_fields else np.array(v))
            for k, v in all_data.items()
        }
        np.savez_compressed(final_output_file, **arrays)
        print(f" [OK] Final merged dataset saved: {final_output_file}")
    except Exception as e:
        print(f" [!] Merge error: {e}")

    return all_data


FEATURE_FIELDS = [
    "name",
    "label",
    "api",
    "pe_imports",
    "pe_sections",
    "signatures",
    "is_running",
    "has_integrity",
]
FEATURE_OBJECT_FIELDS = {"api", "pe_imports", "pe_sections", "signatures"}

SYSCALL_FIELDS = [
    "name",
    "label",
    "is_running",
    "has_integrity",
    "trace_count",
    "trace_data",
]
SYSCALL_OBJECT_FIELDS = {"trace_data"}

# ==============================================================================
# 6. CAPE HTTP INTERACTION
# ==============================================================================
FEATURE_SUBMIT_OPTS = {
    "timeout": 300,
    "platform": "windows",
    "enforce_timeout": False,
    "priority": 1,
    "options": "sniffer=0,procmemdump=0,dumpprocess=0,dump_r0=0,curtain=0,sysmon=0",
}

SYSCALL_SUBMIT_OPTS = {
    "timeout": 60,
    "platform": "windows",
    "machine": "win10",
    "package": "binsim",
    "enforce_timeout": True,
    "priority": 1,
    "options": "sniffer=0,procmemdump=0,dumpprocess=0,dump_r0=0,curtain=0,sysmon=0",
}


def get_headers(server):
    return {"Authorization": f"Token {server['api_token']}"}


def submit_to_cape(file_path, server, submit_opts, request_timeout=30):
    url = f"{server['cape_host']}/apiv2/tasks/create/file/"
    try:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            r = requests.post(
                url,
                headers=get_headers(server),
                files=files,
                data=submit_opts,
                timeout=request_timeout,
            )
            if r.status_code == 200:
                task_ids = r.json().get("data", {}).get("task_ids", [])
                if task_ids:
                    return task_ids[0]
            else:
                print(
                    f" [!] API Submission failed on {server['name']} (HTTP {r.status_code})"
                )
            return None
    except Exception as e:
        print(f" [!] API Connection failed on {server['name']}: {e}")
        return None


def wait_for_report(
    task_id, server, max_polls=240, poll_interval=10, report_timeout=60
):
    status_url = f"{server['cape_host']}/apiv2/tasks/view/{task_id}/"
    report_url = f"{server['cape_host']}/apiv2/tasks/get/report/{task_id}/json/"

    for _ in range(max_polls):
        try:
            r = requests.get(status_url, headers=get_headers(server), timeout=10)
            if r.status_code == 200:
                status = r.json().get("data", {}).get("status")
                if status in ["reported", "failed_analysis", "timeout", "completed"]:
                    rep_req = requests.get(
                        report_url, headers=get_headers(server), timeout=report_timeout
                    )
                    if rep_req.status_code == 200:
                        report_data = rep_req.json()
                        if report_data.get("error") is True:
                            time.sleep(min(poll_interval, 5))
                            continue
                        return report_data
            time.sleep(poll_interval)
        except Exception:
            pass
    return {}


def delete_task(task_id, server):
    url = f"{server['cape_host']}/apiv2/tasks/delete/{task_id}/"
    try:
        requests.get(url, headers=get_headers(server), timeout=10)
    except Exception:
        pass


# ==============================================================================
# 7. PHASE 1 - FEATURE EXTRACTION
# ==============================================================================
def extract_features_raw(report):
    features = {"api": [], "imports": [], "sections": [], "signatures": []}
    if not report:
        return features

    try:
        all_calls = []
        for proc in report.get("behavior", {}).get("processes", []):
            for call in proc.get("calls", []):
                if isinstance(call, dict) and call.get("api"):
                    all_calls.append(call["api"])
        features["api"] = all_calls
    except Exception as e:
        pass

    try:
        target_pe = report.get("target", {}).get("file", {}).get("pe", {})
        static_pe = report.get("static", {}).get("pe", {})
        pe_node = target_pe if target_pe else static_pe

        raw_imports = pe_node.get("imports", [])
        flat_imp = []
        if isinstance(raw_imports, dict):
            for d in raw_imports.values():
                for fn in d.get("imports", []):
                    if fn.get("name"):
                        flat_imp.append(fn["name"])
        elif isinstance(raw_imports, list):
            for d in raw_imports:
                for fn in d.get("imports", []):
                    if fn.get("name"):
                        flat_imp.append(fn["name"])
        features["imports"] = flat_imp[:1000]
        features["sections"] = pe_node.get("sections", [])
    except Exception as e:
        pass

    try:
        for sig in report.get("signatures", []):
            if "name" in sig:
                features["signatures"].append(sig["name"])
    except Exception as e:
        pass

    return features


def process_single_file_features(args):
    file_path, label, server = args
    filename = os.path.basename(file_path)
    start_time = time.time()

    tid = submit_to_cape(file_path, server, FEATURE_SUBMIT_OPTS, request_timeout=30)
    if not tid:
        return {
            "filename": filename,
            "label": label,
            "success": False,
            "server": server["name"],
        }

    print(f" [*] [{server['name']}] Analyzing {filename} ...")

    # REDUCED MAX POLLS: 60 polls * 10 sec = 10 minutes max wait instead of 40 mins
    report = wait_for_report(
        tid, server, max_polls=60, poll_interval=10, report_timeout=60
    )
    feats = extract_features_raw(report)
    delete_task(tid, server)

    duration = time.time() - start_time
    is_running = is_sample_running(report, aux_count=len(feats["api"]))
    has_integrity = check_integrity(report, duration)

    return {
        "filename": filename,
        "label": label,
        "success": True,
        "server": server["name"],
        "duration": duration,
        "is_running": is_running,
        "has_integrity": has_integrity,
        "api": feats["api"],
        "pe_imports": feats["imports"],
        "pe_sections": feats["sections"],
        "signatures": feats["signatures"],
    }


def feature_extra_fields(res):
    return {
        "api": res["api"],
        "pe_imports": res["pe_imports"],
        "pe_sections": res["pe_sections"],
        "signatures": res["signatures"],
    }


# ==============================================================================
# 8. PHASE 2 - SYSCALL / BINSIM TRACE EXTRACTION
# ==============================================================================
def find_binsim_sha256(obj):
    if isinstance(obj, dict):
        name = obj.get("name") or obj.get("filepath") or obj.get("path")
        if name and "binsim_trace.json" in str(name) and "sha256" in obj:
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


def extract_binsim_trace(report, task_id, server):
    assembly_slices = []
    sha256 = find_binsim_sha256(report)
    if not sha256:
        return assembly_slices

    dl_url = f"{server['cape_host']}/apiv2/tasks/get/dropped/{task_id}/"
    try:
        r = requests.get(dl_url, headers=get_headers(server), timeout=45)
        if r.status_code == 200:
            file_bytes = io.BytesIO(r.content)
            extracted_text = None
            with pyzipper.AESZipFile(file_bytes) as z:
                z.setpassword(b"infected")
                for name in z.namelist():
                    if name.endswith(".json") and (
                        sha256 in name or "binsim_trace" in name
                    ):
                        extracted_text = z.read(name).decode("utf-8", errors="ignore")
                        break
            if extracted_text:
                for line in extracted_text.strip().split("\n"):
                    if line:
                        try:
                            assembly_slices.append(json.loads(line))
                        except Exception:
                            pass
    except Exception:
        pass
    return assembly_slices


def process_single_file_syscall(args):
    file_path, label, server = args
    filename = os.path.basename(file_path)
    start_time = time.time()

    tid = submit_to_cape(file_path, server, SYSCALL_SUBMIT_OPTS, request_timeout=35)
    if not tid:
        return {
            "filename": filename,
            "label": label,
            "success": False,
            "server": server["name"],
        }

    print(f" [*] [{server['name']}] Tracing syscalls for {filename} ...")

    report = wait_for_report(
        tid, server, max_polls=120, poll_interval=2, report_timeout=45
    )
    traces = extract_binsim_trace(report, tid, server)
    delete_task(tid, server)

    duration = time.time() - start_time
    trace_count = len(traces)
    is_running = is_sample_running(report, aux_count=trace_count)
    has_integrity = check_integrity(report, duration)

    if not has_integrity:
        try:
            now = datetime.now()
            stamp = (
                f"{now.year}{now.month:02d}{now.day:02d}_{now.hour:02d}{now.minute:02d}"
            )
            shutil.copyfile(
                file_path, os.path.join(FAIL_PATH, f"corrupt_{stamp}_{filename}")
            )
        except Exception:
            pass

    return {
        "filename": filename,
        "label": label,
        "success": True,
        "server": server["name"],
        "duration": duration,
        "is_running": is_running,
        "has_integrity": has_integrity,
        "trace_count": trace_count,
        "trace_data": traces,
    }


def syscall_extra_fields(res):
    return {"trace_count": res["trace_count"], "trace_data": res["trace_data"]}


# ==============================================================================
# 9. SSH HARD CLEANUP
# ==============================================================================
def hard_clean_cape_server(server):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            server["ssh_host"],
            username=server["ssh_user"],
            password=server["ssh_pass"],
            timeout=10,
        )

        commands = [
            'echo "db.dropDatabase()" | mongo cape || echo "db.dropDatabase()" | mongosh cape',
            f'echo "{server["ssh_pass"]}" | sudo -S rm -rf {server["cape_dir"]}/storage/analyses/*',
            f'echo "{server["ssh_pass"]}" | sudo -S rm -rf {server["cape_dir"]}/storage/binaries/*',
            f'echo "{server["ssh_pass"]}" | sudo -S systemctl restart cape.service',
            f'echo "{server["ssh_pass"]}" | sudo -S systemctl restart cape-processor.service',
            f'echo "{server["ssh_pass"]}" | sudo -S systemctl restart cape-web.service',
        ]
        for cmd in commands:
            _, stdout, _ = ssh.exec_command(cmd)
            stdout.channel.recv_exit_status()
        ssh.close()
    except Exception as e:
        print(f" [!] SSH cleanup error on {server['name']}: {e}")


def clean_all_servers():
    print(
        "\n [SSH] Hard cleanup across all CAPE servers (DB + storage + service restart)..."
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(SERVERS)) as executor:
        executor.map(hard_clean_cape_server, SERVERS)

    # REVERTED BACK TO 10 SECONDS AS REQUESTED
    print(" [SSH] Cleanup complete. Waiting 10s for services to stabilize...\n")
    time.sleep(10)


# ==============================================================================
# 10. WAVE RUNNER (UPDATED TO YIELD IMMEDIATELY ON COMPLETION)
# ==============================================================================
def run_extraction_phase(
    phase_label,
    dataset_name,
    target_dir,
    workspace_dir,
    output_npz,
    worker_fn,
    fields,
    object_fields,
    extra_field_builder,
):
    ensure_dir(workspace_dir)
    processed_set = load_json_set(os.path.join(workspace_dir, "processed_log.json"))

    files_to_process = []
    for root, _, files in os.walk(target_dir):
        for fname in files:
            if fname.startswith(".") or fname in processed_set:
                continue
            files_to_process.append((os.path.join(root, fname), dataset_name))

    total = len(files_to_process)
    print(
        f"--- [{phase_label}] {len(processed_set)} already done, {total} remaining in {target_dir} ---"
    )
    if total == 0:
        return

    current_batch = {k: [] for k in fields}
    integrity_updates = {}
    done = 0
    chunk_size = len(SERVERS)

    try:
        for i in range(0, total, chunk_size):
            chunk = files_to_process[i : i + chunk_size]
            tasks = [
                (fp, lbl, SERVERS[j % len(SERVERS)])
                for j, (fp, lbl) in enumerate(chunk)
            ]

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=chunk_size
            ) as executor:
                # AS_COMPLETED: Prints and handles results instantly!
                futures = [executor.submit(worker_fn, task) for task in tasks]
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()

                    if not res.get("success"):
                        print(
                            f" [!] SKIPPED {res['filename']} - Could not process on {res.get('server', 'unknown')}"
                        )
                        continue

                    done += 1
                    pct = done / total * 100
                    if "trace_count" in res:
                        extra_info = f"Traces: {res['trace_count']}"
                    else:
                        extra_info = (
                            f"API: {len(res['api'])} | Imp: {len(res['pe_imports'])} | "
                            f"Sec: {len(res['pe_sections'])} | Sig: {len(res['signatures'])}"
                        )
                    status = "INTACT" if res["has_integrity"] else "CORRUPT"

                    # LOGS EXACTLY WHEN IT FINISHES
                    print(
                        f"   [{pct:05.1f}% | {done}/{total}] [{res['server']}] {res['filename']} "
                        f"-> {status} | {extra_info} ({res['duration']:.1f}s)"
                    )

                    current_batch["name"].append(res["filename"])
                    current_batch["label"].append(res["label"])
                    current_batch["is_running"].append(res["is_running"])
                    current_batch["has_integrity"].append(res["has_integrity"])
                    for k, v in extra_field_builder(res).items():
                        current_batch[k].append(v)

                    integrity_updates[res["filename"]] = {
                        "label": res["label"],
                        "is_running": res["is_running"],
                        "has_integrity": res["has_integrity"],
                        "alive": bool(res["is_running"] and res["has_integrity"]),
                    }

            if len(current_batch["name"]) >= BATCH_SIZE:
                save_batch_npz(current_batch, workspace_dir, object_fields)
                save_integrity_state(workspace_dir, integrity_updates)
                integrity_updates = {}
                clean_all_servers()
                current_batch = {k: [] for k in fields}

    except KeyboardInterrupt:
        print(f"\n [STOP] {phase_label} halted by user.")
    except Exception as e:
        print(f"\n [CRASH] {phase_label} error: {e}")
    finally:
        if current_batch["name"]:
            print(f"\n [SAFETY SAVE] {len(current_batch['name'])} remaining samples...")
            save_batch_npz(current_batch, workspace_dir, object_fields)
            save_integrity_state(workspace_dir, integrity_updates)
            clean_all_servers()

        merge_all_npz(workspace_dir, output_npz, fields, object_fields)


# ==============================================================================
# 11. COMBINED INTEGRITY REPORT
# ==============================================================================
def build_integrity_report(dataset_name, feat_state, sys_state, report_path):
    all_filenames = set(feat_state.keys()) | set(sys_state.keys())
    samples = []
    alive_count = 0

    for fname in sorted(all_filenames):
        f1 = feat_state.get(fname)
        f2 = sys_state.get(fname)
        phase1_alive = bool(f1["alive"]) if f1 else False
        phase2_alive = bool(f2["alive"]) if f2 else False
        final_alive = phase1_alive or phase2_alive
        if final_alive:
            alive_count += 1

        samples.append(
            {
                "name": fname,
                "label": (f1 or f2 or {}).get("label"),
                "phase1_feature_extraction": {
                    "processed": f1 is not None,
                    "is_running": f1["is_running"] if f1 else None,
                    "has_integrity": f1["has_integrity"] if f1 else None,
                    "alive": phase1_alive,
                },
                "phase2_syscall_extraction": {
                    "processed": f2 is not None,
                    "is_running": f2["is_running"] if f2 else None,
                    "has_integrity": f2["has_integrity"] if f2 else None,
                    "alive": phase2_alive,
                },
                "final_status": "alive" if final_alive else "dead",
            }
        )

    total = len(samples)
    dead_count = total - alive_count
    report = {
        "dataset": dataset_name,
        "generated_at": datetime.now().isoformat(),
        "total_samples": total,
        "alive_count": alive_count,
        "dead_count": dead_count,
        "alive_pct": round(alive_count / total * 100, 2) if total else 0.0,
        "samples": samples,
    }

    with codecs.open(report_path, "w", "utf-8") as f:
        json.dump(report, f, indent=2)

    print(
        f" [REPORT] {dataset_name}: {alive_count}/{total} alive ({report['alive_pct']}%) -> {report_path}"
    )
    return report


# ==============================================================================
# 12. MAIN
# ==============================================================================
def main():
    ensure_dir(FAIL_PATH)

    print("=" * 70)
    print(" ULTIMATE CAPEv2 EXTRACTION PIPELINE")
    print(f" Servers: {len(SERVERS)} | Datasets: {len(DATASETS)}")
    print("=" * 70)

    # ---------------- PHASE 1: ALL datasets, feature extraction ----------------
    print("\n" + "#" * 70)
    print(
        " PHASE 1/2 - STATIC & DYNAMIC FEATURE EXTRACTION (API / Imports / Sections / Signatures)"
    )
    print("#" * 70)
    for dataset_name in DATASETS:
        paths = dataset_paths(dataset_name)
        if not os.path.exists(paths["dir"]):
            print(f" [SKIP] {dataset_name}: directory not found ({paths['dir']})")
            continue
        print(f"\n--- Dataset: {dataset_name} ---")
        run_extraction_phase(
            phase_label=f"{dataset_name}/features",
            dataset_name=dataset_name,
            target_dir=paths["dir"],
            workspace_dir=paths["feat_workspace"],
            output_npz=paths["feat_output"],
            worker_fn=process_single_file_features,
            fields=FEATURE_FIELDS,
            object_fields=FEATURE_OBJECT_FIELDS,
            extra_field_builder=feature_extra_fields,
        )

    # ---------------- PHASE 2: ALL datasets, syscall extraction ----------------
    print("\n" + "#" * 70)
    print(" PHASE 2/2 - SYSCALL / BINSIM TRACE EXTRACTION")
    print("#" * 70)
    for dataset_name in DATASETS:
        paths = dataset_paths(dataset_name)
        if not os.path.exists(paths["dir"]):
            continue
        print(f"\n--- Dataset: {dataset_name} ---")
        run_extraction_phase(
            phase_label=f"{dataset_name}/syscall",
            dataset_name=dataset_name,
            target_dir=paths["dir"],
            workspace_dir=paths["sys_workspace"],
            output_npz=paths["sys_output"],
            worker_fn=process_single_file_syscall,
            fields=SYSCALL_FIELDS,
            object_fields=SYSCALL_OBJECT_FIELDS,
            extra_field_builder=syscall_extra_fields,
        )

    # ---------------- FINAL: combined dead/alive report per dataset ----------------
    print("\n" + "#" * 70)
    print(" BUILDING COMBINED INTEGRITY REPORTS")
    print("#" * 70)
    dataset_reports = {}
    for dataset_name in DATASETS:
        paths = dataset_paths(dataset_name)
        if not os.path.exists(paths["dir"]):
            continue
        feat_state = load_integrity_state(paths["feat_workspace"])
        sys_state = load_integrity_state(paths["sys_workspace"])
        if not feat_state and not sys_state:
            continue
        dataset_reports[dataset_name] = build_integrity_report(
            dataset_name, feat_state, sys_state, paths["report_output"]
        )

    grand_total = sum(r["total_samples"] for r in dataset_reports.values())
    grand_alive = sum(r["alive_count"] for r in dataset_reports.values())

    print("\n" + "=" * 75)
    print(" RUN SUMMARY")
    print("=" * 75)
    print(
        f"{'DATASET':<15} | {'TOTAL':<8} | {'ALIVE':<8} | {'DEAD':<8} | {'ALIVE %':<8}"
    )
    print("-" * 75)
    for ds_name, rep in dataset_reports.items():
        print(
            f"{ds_name.upper():<15} | {rep['total_samples']:<8} | {rep['alive_count']:<8} | "
            f"{rep['dead_count']:<8} | {rep['alive_pct']:<8.1f}"
        )
    print("-" * 75)
    if grand_total:
        print(
            f"OVERALL: {grand_alive}/{grand_total} alive ({grand_alive / grand_total * 100:.1f}%)"
        )
    print("=" * 75)


if __name__ == "__main__":
    main()
