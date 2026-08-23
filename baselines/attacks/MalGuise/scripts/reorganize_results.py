#!/usr/bin/env python3
import os
import glob
import json
import shutil
from pathlib import Path

def main():
    base_dir = Path("results")
    if not base_dir.exists():
        print("No results directory found!")
        return

    # Define new paths
    adv_dir = base_dir / "adversarial" / "adv"
    info_dir = base_dir / "adversarial" / "info"
    fail_dir = base_dir / "fail"
    
    # Create new directories
    adv_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)

    # We will iterate over all sample folders in results/ (excluding _logs, adversarial, fail)
    families = [d for d in base_dir.iterdir() if d.is_dir() and d.name not in ["_logs", "adversarial", "fail"]]
    
    moved_success = 0
    moved_fail = 0

    for family_dir in families:
        for sample_dir in family_dir.iterdir():
            if not sample_dir.is_dir():
                continue
                
            sample_id = sample_dir.name
            result_json = sample_dir / f"{sample_id}_adv_result.json"
            failure_json = sample_dir / f"{sample_id}_adv_failure.json"
            adv_exe = sample_dir / f"{sample_id}_adv.exe"

            is_success = False

            # Check if it was a successful evasion
            if result_json.exists():
                try:
                    with open(result_json, "r") as f:
                        data = json.load(f)
                        if data.get("success") == True:
                            is_success = True
                except:
                    pass

            if is_success:
                # To match VT CLI tool format, group by family inside adv_dir
                family_adv_dir = adv_dir / family_dir.name
                family_adv_dir.mkdir(parents=True, exist_ok=True)
                
                # Move to adversarial
                if adv_exe.exists():
                    shutil.move(str(adv_exe), str(family_adv_dir / f"{sample_id}_adv.exe"))
                
                # Move ALL json files (result/final_cfg etc) to info/ (also grouped by family)
                family_info_dir = info_dir / family_dir.name
                family_info_dir.mkdir(parents=True, exist_ok=True)
                for jfile in sample_dir.glob("*.json"):
                    shutil.move(str(jfile), str(family_info_dir / jfile.name))
                    
                moved_success += 1
            else:
                # Move entire sample folder to fail/
                dest_fail = fail_dir / sample_id
                if dest_fail.exists():
                    shutil.rmtree(dest_fail)
                shutil.move(str(sample_dir), str(dest_fail))
                moved_fail += 1

            # Remove the empty sample directory if it still exists (in success case)
            if sample_dir.exists() and not any(sample_dir.iterdir()):
                sample_dir.rmdir()

        # Remove the family directory if empty
        if not any(family_dir.iterdir()):
            family_dir.rmdir()

    print(f"Reorganization complete!")
    print(f" - Successfully evaded samples moved to: {adv_dir} ({moved_success} samples)")
    print(f" - Failed samples moved to: {fail_dir} ({moved_fail} samples)")

if __name__ == "__main__":
    main()
