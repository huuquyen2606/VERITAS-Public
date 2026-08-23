import redis
import json
import time

try:
    import fast_sw
except ImportError:
    print(" [!] Error: Module 'fast_sw' not found. Build the C++ extension with 'pip install .'")
    exit(1)

from sequence_deduplicator import deduplicate_sequence

# System configuration
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0

# List of critical APIs
CRITICAL_APIS = {
    # Process Operations
    "NtCreateProcess", "NtOpenProcess", "NtTerminateProcess",
    
    # Thread Operations
    "NtCreateThread", "NtResumeThread", "NtTerminateThread",
    
    # File Operations
    "NtCreateFile", "NtOpenFile", "NtClose", 
    "NtQueryDirectoryFile", "NtSetInformationFile",
    
    # Registry Operations
    "NtCreateKey", "NtOpenKey", "NtSaveKey",
    
    # Memory Operations
    "NtAllocateVirtualMemory", "NtMapViewOfSection", "NtWriteVirtualMemory",
    
    # Network Operations
    "connect", "bind", "send", "recv", "gethostname",
    
    # Desktop Operations
    "CreateDesktop", "SwitchDesktop", "SetThreadDesktop",
    
    # Other Core Operations
    "LoadLibrary", "GetProcAddress", "GetModuleHandle"
}

# Redis connection
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)


def convert_to_apicall_objects(raw_sequence):
    """
    Convert raw API sequence from sandbox report into fast_sw.ApiCall C++ objects.
    """
    api_objects = []
    for item in raw_sequence:
        name = ""
        attributes = ""
        
        if isinstance(item, dict):
            name = item.get("name", item.get("api", ""))
            args = item.get("arguments", {})
            if isinstance(args, dict):
                attributes = "|".join([f"{k}:{v}" for k, v in args.items()])
            elif isinstance(args, list):
                attributes = "|".join([str(x) for x in args])
        elif isinstance(item, str):
            name = item
            
        if not name:
            continue
            
        is_critical = name in CRITICAL_APIS
        api_obj = fast_sw.ApiCall(name, attributes, is_critical)
        api_objects.append(api_obj)
        
    return api_objects


def evaluate_api_similarity(episode_id, original_name, adv_name):
    """
    End-to-end API similarity evaluation pipeline.
    """
    print(f"\n[*] Evaluating API similarity for episode: {episode_id}")
    
    baseline_key = f"baseline:{original_name}:data"
    episode_key = f"episode:{adv_name}:data"
    
    baseline_raw = r.get(baseline_key)
    episode_raw = r.get(episode_key)
    
    if not baseline_raw or not episode_raw:
        print(f" [!] Missing data for {original_name} or {adv_name}")
        return 0.0
        
    baseline_data = json.loads(baseline_raw)
    episode_data = json.loads(episode_raw)
    
    seq_ori = baseline_data.get("api_chain", [])
    seq_adv = episode_data.get("api_chain", [])
    
    if not seq_ori:
        print(" [!] Original sample has no API data. Skipping comparison.")
        return 0.0

    start_time = time.time()

    clean_ori = deduplicate_sequence(seq_ori, lm=5, k=2)
    clean_adv = deduplicate_sequence(seq_adv, lm=5, k=2)
    
    c_ori = convert_to_apicall_objects(clean_ori)
    c_adv = convert_to_apicall_objects(clean_adv)

    similarity_score = fast_sw.calculate_similarity(c_ori, c_adv)
    
    exec_time = time.time() - start_time
    print(f"[*] Result: {similarity_score:.2f}% (processed in {exec_time:.4f}s)")

    episode_data["functionality_api_score"] = similarity_score
    r.set(episode_key, json.dumps(episode_data))
    
    return similarity_score