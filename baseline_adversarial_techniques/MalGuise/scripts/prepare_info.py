import os
import json
import glob
import shutil
from pathlib import Path

def prepare_info_folder():
    base_dir = Path(__file__).resolve().parent.parent
    results_dir = base_dir / "results"
    info_dir = base_dir / "AEs_MalGuise_Info"
    vt_reports_src = base_dir.parent.parent / "AVs" / "VirusTotal-CLI-Tool" / "vt_reports"
    
    # Target dirs
    cfg_dest = info_dir / "CFG"
    adv_dest = info_dir / "AEs_MalGuise"
    vt_dest = info_dir / "vt_reports"
    
    cfg_dest.mkdir(parents=True, exist_ok=True)
    adv_dest.mkdir(parents=True, exist_ok=True)
    vt_dest.mkdir(parents=True, exist_ok=True)
    
    successes = []
    for f in glob.glob(str(results_dir / "*" / "*" / "*_result.json")):
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                if data.get('success'):
                    successes.append(Path(f))
        except:
            pass

    print(f"Found {len(successes)} successful samples in results/")
    
    copied_cfg = 0
    copied_adv = 0
    copied_vt = 0
    
    for res_json in successes:
        sample_dir = res_json.parent
        family = sample_dir.parent.name
        sample_id = res_json.name.replace("_adv_result.json", "")
        
        # 1. Copy CFG
        cfg_src = sample_dir / f"{sample_id}_adv_final_cfg.json"
        if cfg_src.exists():
            shutil.copy2(cfg_src, cfg_dest / f"{sample_id}_adv_final_cfg.json")
            copied_cfg += 1
            
        # 2. Copy ADV EXE
        adv_src = sample_dir / f"{sample_id}_adv.exe"
        if adv_src.exists():
            # Group by family in destination to match the standard
            family_adv_dest = adv_dest / family
            family_adv_dest.mkdir(exist_ok=True)
            shutil.copy2(adv_src, family_adv_dest / f"{sample_id}_adv.exe")
            copied_adv += 1
            
        # 3. Copy VT Report
        # VT reports are in vt_reports/<family>/<sample_id>_adv.exe.json
        vt_src = vt_reports_src / family / f"{sample_id}_adv.exe.json"
        if vt_src.exists():
            family_vt_dest = vt_dest / family
            family_vt_dest.mkdir(exist_ok=True)
            shutil.copy2(vt_src, family_vt_dest / f"{sample_id}_adv.exe.json")
            copied_vt += 1
            
    print(f"Copied {copied_adv} adversarial executables.")
    print(f"Copied {copied_cfg} CFG files.")
    print(f"Copied {copied_vt} VirusTotal reports.")
    
    # Validate final counts
    total_cfg = len(list(cfg_dest.glob("*.json")))
    total_adv = len(list(adv_dest.rglob("*.exe")))
    total_vt = len(list(vt_dest.rglob("*.json")))
    
    print("\n--- SUMMARY ---")
    print(f"Total in AEs_MalGuise: {total_adv} / 72")
    print(f"Total in CFG: {total_cfg} / 72")
    print(f"Total in vt_reports: {total_vt} / 72")
    
if __name__ == "__main__":
    prepare_info_folder()
