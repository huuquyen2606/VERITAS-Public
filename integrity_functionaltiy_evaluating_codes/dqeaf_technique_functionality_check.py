import os
import json
import angr
import logging
import argparse

# --- DEFAULT CONFIGURATION ---
DEFAULT_ORI_DIR = "Test"
DEFAULT_ADV_DIRS = ["malguise"]
DEFAULT_REPORT_PATH = "malguise_preservation_report.json"

# Suppress angr's extremely verbose logging so our console stays clean
# Note: You will still see one 'unicornlib' warning at the very start.
# This is completely safe to ignore since we only use static analysis (CFGFast).
logging.getLogger("angr").setLevel(logging.CRITICAL)
logging.getLogger("cle").setLevel(logging.CRITICAL)


def get_all_files(directory):
    """
    Recursively walks through a directory and returns a list of full file paths.
    """
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def extract_cfg_stats(filepath):
    """
    Uses angr to generate CFGFast and returns the number of nodes and edges.
    """
    try:
        # auto_load_libs=False is CRITICAL for speed. We only care about the main binary.
        proj = angr.Project(filepath, load_options={"auto_load_libs": False})

        # Generate the fast Control Flow Graph (static analysis only)
        cfg = proj.analyses.CFGFast()

        # FIXED: Wrap the generator in list() so Python can count the length!
        nodes_count = len(list(cfg.graph.nodes()))
        edges_count = len(list(cfg.graph.edges()))

        return nodes_count, edges_count
    except Exception as e:
        print(f" [!] angr failed to analyze {filepath}: {e}")
        return None, None


def evaluate_multiple_directories(ori_dir, adv_dirs, report_path):
    print("--- STARTING MULTI-DIRECTORY EVALUATION ---")

    if not os.path.exists(ori_dir):
        print(f" [!] Error: Original directory '{ori_dir}' not found.")
        return

    # Grab all actual files recursively from the Test directory
    ori_file_paths = get_all_files(ori_dir)
    total_ori = len(ori_file_paths)
    print(f" [INFO] Original samples found: {total_ori}")

    # Master dictionary to hold the final JSON structure
    report_data = {
        "global_summary": {
            "total_original_samples": total_ori,
            "directories_evaluated": len(adv_dirs),
        },
        "results_by_directory": {},
    }

    # Loop through each adversarial directory provided in the list
    for adv_dir in adv_dirs:
        print(f"\n====================================================")
        print(f"--- EVALUATING DIRECTORY: {adv_dir.upper()} ---")
        print(f"====================================================")

        if not os.path.exists(adv_dir):
            print(f" [!] Warning: Directory '{adv_dir}' not found. Skipping.")
            continue

        # Grab all actual files recursively from the current adversarial directory
        adv_file_paths = get_all_files(adv_dir)
        total_adv = len(adv_file_paths)

        matched_count = 0
        preserve_count = 0
        broken_count = 0
        dir_details = []

        for adv_path in adv_file_paths:
            # Extract just the filename (e.g., 'adv_sample1.exe') from the full path
            adv_name = os.path.basename(adv_path)

            matched_ori_path = None
            matched_ori_name = None

            # String matching: Check if the original base name is a subset of the adversarial name
            for ori_path in ori_file_paths:
                ori_name = os.path.basename(ori_path)
                # Strip the extension to make matching more robust (e.g., 'sample1' in 'adv_sample1.exe')
                ori_base = os.path.splitext(ori_name)[0]

                if ori_base in adv_name:
                    matched_ori_path = ori_path
                    matched_ori_name = ori_name
                    break

            if matched_ori_path:
                matched_count += 1

                print(
                    f" [*] [{adv_dir}] Comparing: {matched_ori_name} -> {adv_name}..."
                )

                # Extract CFG stats using the FULL paths
                ori_nodes, ori_edges = extract_cfg_stats(matched_ori_path)
                adv_nodes, adv_edges = extract_cfg_stats(adv_path)

                # Determine preservation (Strict 100% match required for CFG structural integrity)
                is_preserved = False
                if (
                    (ori_nodes is not None and adv_nodes is not None)
                    and (ori_nodes == adv_nodes)
                    and (ori_edges == adv_edges)
                ):
                    is_preserved = True
                    preserve_count += 1
                else:
                    broken_count += 1

                # Log details for this specific file pair
                dir_details.append(
                    {
                        "original_file": matched_ori_name,
                        "adversarial_file": adv_name,
                        "status": "preserved" if is_preserved else "broken",
                        "stats": {
                            "original_nodes": ori_nodes,
                            "original_edges": ori_edges,
                            "adversarial_nodes": adv_nodes,
                            "adversarial_edges": adv_edges,
                        },
                    }
                )
            else:
                print(
                    f" [!] Warning: No original match found for {adv_name} in {adv_dir}"
                )

        # Calculate metrics for this specific directory
        rate_matched = (
            (preserve_count / matched_count * 100) if matched_count > 0 else 0.0
        )
        rate_original = (preserve_count / total_ori * 100) if total_ori > 0 else 0.0

        print(f"\n --- METRICS FOR: {adv_dir} ---")
        print(
            f" Matches: {matched_count} | Preserved: {preserve_count} | Broken: {broken_count}"
        )
        print(f" Similarity Rate (vs MATCHED)  : {rate_matched:.2f}%")
        print(f" Similarity Rate (vs ORIGINAL) : {rate_original:.2f}%")

        # Append this directory's results to the master JSON structure
        report_data["results_by_directory"][adv_dir] = {
            "summary": {
                "total_adversarial_samples": total_adv,
                "matched_pairs": matched_count,
                "functionality_preserved": preserve_count,
                "functionality_broken": broken_count,
                "preservation_rate_vs_matched_percent": round(rate_matched, 2),
                "preservation_rate_vs_original_percent": round(rate_original, 2),
            },
            "details": dir_details,
        }

    # ==========================================
    # SAVE CONSOLIDATED JSON REPORT
    # ==========================================
    print("\n================ FINAL REPORT ===================")
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=4)

    print(f" [*] Detailed JSON report successfully saved to '{report_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DQEAF technique functionality preservation.")
    parser.add_argument("--ori-dir", default=DEFAULT_ORI_DIR, help="Directory containing original samples.")
    parser.add_argument("--adv-dirs", nargs="+", default=DEFAULT_ADV_DIRS, help="One or more directories containing adversarial samples.")
    parser.add_argument("--report", default=DEFAULT_REPORT_PATH, help="Output JSON report path.")
    
    args = parser.parse_args()
    
    evaluate_multiple_directories(args.ori_dir, args.adv_dirs, args.report)
