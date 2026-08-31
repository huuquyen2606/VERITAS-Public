import os
import time
import requests
import numpy as np

# --- CUCKOO CONFIG (Primary) ---
CUCKOO_HOST = "http://192.168.2.123:8090"
CUCKOO_TOKEN = "4tnVmJAddAjA7AUuGR6yvA"
CUCKOO_HEADERS = {"Authorization": f"Bearer {CUCKOO_TOKEN}"}

# --- CAPE CONFIG (Fallback) ---
CAPE_HOST = "http://192.168.2.118:8000"
CAPE_TOKEN = "5f42b8083e6a6e95bd32ff2037ae323252dcb8ff"
CAPE_HEADERS = {"Authorization": f"Token {CAPE_TOKEN}"}


# ==========================================
# CUCKOO LOGIC
# ==========================================
def submit_cuckoo(file_path):
    url = f"{CUCKOO_HOST}/tasks/create/file"
    options = "procmemdump=0,dumpprocess=0,memory_dump=0,sniffer=0"
    data = {"timeout": 300, "enforce_timeout": False, "priority": 1, "options": options}
    
    try:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            r = requests.post(url, headers=CUCKOO_HEADERS, files=files, data=data, timeout=30)
            if r.status_code == 200:
                return r.json().get("task_id")
    except:
        pass
    return None

def wait_cuckoo(task_id):
    status_url = f"{CUCKOO_HOST}/tasks/view/{task_id}"
    report_url = f"{CUCKOO_HOST}/tasks/report/{task_id}"
    print(f"   [+] Cuckoo Task {task_id}...", end="", flush=True)
    
    for i in range(240):
        try:
            r = requests.get(status_url, headers=CUCKOO_HEADERS, timeout=10)
            if r.status_code == 200:
                status = r.json().get("task", {}).get("status")
                if status in ["reported", "failed_analysis", "completed"]:
                    rep = requests.get(report_url, headers=CUCKOO_HEADERS, timeout=60)
                    if rep.status_code == 200:
                        print(" OK!")
                        return rep.json()
            time.sleep(5)
            if i % 12 == 0:
                print(".", end="", flush=True)
        except:
            time.sleep(5)
    print(" Timeout!")
    return {}

def delete_cuckoo(task_id):
    url = f"{CUCKOO_HOST}/tasks/delete/{task_id}"
    try: requests.get(url, headers=CUCKOO_HEADERS, timeout=10)
    except: pass

def extract_cuckoo(report):
    features = {"api": [], "imports": [], "sections": []}
    if not report: return features
    
    try: 
        for proc in report.get("behavior", {}).get("processes", []):
            for call in proc.get("calls", []):
                if "api" in call:
                    features["api"].append(call["api"])
    except: pass
    
    try: 
        imp_raw = report.get("static", {}).get("pe_imports", [])
        flat_imp = []
        if isinstance(imp_raw, list):
            for d in imp_raw:
                for f in d.get("imports", []):
                    if f.get("name"): flat_imp.append(f["name"])
        elif isinstance(imp_raw, dict):
            for d in imp_raw.values():
                for f in d.get("imports", []):
                    if f.get("name"): flat_imp.append(f["name"])
        features["imports"] = flat_imp[:1000]
    except: pass
    
    try: 
        sec = report.get("static", {}).get("pe_sections", [])
        if not sec:
            sec = report.get("static", {}).get("pe", {}).get("sections", [])
        features["sections"] = sec
    except: pass
    return features


# ==========================================
# CAPE LOGIC (Fallback)
# ==========================================
def submit_cape(file_path):
    url = f"{CAPE_HOST}/apiv2/tasks/create/file/"
    options_str = "sniffer=0,procmemdump=0,dumpprocess=0,dump_r0=0,curtain=0,sysmon=0"
    data_params = {"timeout": 300, "platform": "windows", "enforce_timeout": False, "options": options_str}
    
    try:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            r = requests.post(url, headers=CAPE_HEADERS, files=files, data=data_params, timeout=30)
            if r.status_code == 200:
                task_ids = r.json().get("data", {}).get("task_ids", [])
                if task_ids: return task_ids[0]
    except:
        pass
    return None

def wait_cape(task_id):
    status_url = f"{CAPE_HOST}/apiv2/tasks/view/{task_id}/"
    report_url = f"{CAPE_HOST}/apiv2/tasks/get/report/{task_id}/json/"
    print(f"   [+] CAPE Task {task_id}...", end="", flush=True)
    
    for i in range(240):
        try:
            r = requests.get(status_url, headers=CAPE_HEADERS, timeout=10)
            if r.status_code == 200:
                status = r.json().get("data", {}).get("status")
                if status in ["reported", "failed_analysis", "timeout", "completed"]:
                    rep = requests.get(report_url, headers=CAPE_HEADERS, timeout=60)
                    if rep.status_code == 200:
                        report_data = rep.json()
                        if report_data.get("error") is True:
                            time.sleep(5)
                            continue
                        print(" OK!")
                        return report_data
            time.sleep(5)
            if i % 12 == 0:
                print(".", end="", flush=True)
        except:
            time.sleep(5)
    print(" Timeout!")
    return {}

def delete_cape(task_id):
    url = f"{CAPE_HOST}/apiv2/tasks/delete/{task_id}/"
    try: requests.get(url, headers=CAPE_HEADERS, timeout=10)
    except: pass

def extract_cape(report):
    features = {"api": [], "imports": [], "sections": []}
    if not report: return features
    
    try:
        for proc in report.get("behavior", {}).get("processes", []):
            for call in proc.get("calls", []):
                if isinstance(call, dict) and call.get("api"):
                    features["api"].append(call["api"])
    except: pass
    
    try:
        target_pe = report.get("target", {}).get("file", {}).get("pe", {})
        static_pe = report.get("static", {}).get("pe", {})
        pe_node = target_pe if target_pe else static_pe
        raw_imports = pe_node.get("imports", [])
        flat_imp = []
        if isinstance(raw_imports, dict):
            for d in raw_imports.values():
                for f in d.get("imports", []):
                    if f.get("name"): flat_imp.append(f["name"])
        elif isinstance(raw_imports, list):
            for d in raw_imports:
                for f in d.get("imports", []):
                    if f.get("name"): flat_imp.append(f["name"])
        features["imports"] = flat_imp[:1000]
        features["sections"] = pe_node.get("sections", [])
    except: pass
    return features


# ==========================================
# MAIN EXPORTED FUNCTION
# ==========================================
def process_raw_pe_to_npz(file_path, output_npz):
    """
    Submits a PE file dynamically to Cuckoo (or CAPE), extracts features,
    and packages them into the .npz format expected by the dataset loaders.
    """
    tid = submit_cuckoo(file_path)
    if tid:
        report = wait_cuckoo(tid)
        feats = extract_cuckoo(report)
        delete_cuckoo(tid)
    else:
        print(f"   [!] Cuckoo failed. Fallback to CAPE...")
        tid = submit_cape(file_path)
        if tid:
            report = wait_cape(tid)
            feats = extract_cape(report)
            delete_cape(tid)
        else:
            print("   [!] Both servers failed or unreachable.")
            return False

    try:
        with open(file_path, "rb") as f:
            raw_bytes = list(f.read())
    except:
        raw_bytes = []
        
    np.savez_compressed(
        output_npz,
        name=np.array([os.path.basename(file_path)]),
        label=np.array(["Unknown"]),
        api_cuckoo=np.array([feats.get("api", [])], dtype=object),
        pe_imports=np.array([feats.get("imports", [])], dtype=object),
        pe_sections=np.array([feats.get("sections", [])], dtype=object),
        raw_byte=np.array([raw_bytes], dtype=object)
    )
    return True
