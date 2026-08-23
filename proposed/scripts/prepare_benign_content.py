"""Extract benign PE components into a serialized bank for PE mutation actions."""

import os
import pickle
import argparse
import lief

def build_benign_bank(exe_folder: str, output_dir: str):
    print(f"[*] Starting data extraction from directory: {exe_folder}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    pickle_output_file = os.path.join(output_dir, "benign_bank.pkl")

    benign_bank = {
        "code_bytes": [],      
        "data_bytes": [],      
        "resource_bytes": [],  
        "dos_stubs": [],       
        "section_meta": [],    
        "imports_dict": {}     
    }

    valid_files = 0

    for filename in os.listdir(exe_folder):
        filepath = os.path.join(exe_folder, filename)
        if not os.path.isfile(filepath):
            continue

        try:
            pe = lief.PE.parse(filepath)
            if pe is None:
                continue
                
            valid_files += 1
            print(f"[*] Parsing file: {filename}")

            if pe.dos_stub:
                benign_bank["dos_stubs"].append(list(pe.dos_stub))

            for sec in pe.sections:
                sec_name = sec.name.replace('\x00', '')
                chars = sec.characteristics
                content = list(sec.content)
                
                benign_bank["section_meta"].append((sec_name, chars))

                if not content:
                    continue
                    
                if int(chars) & int(lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE):
                    benign_bank["code_bytes"].append(content)
                elif sec_name == ".rsrc":
                    benign_bank["resource_bytes"].append(content)
                else:
                    benign_bank["data_bytes"].append(content)

            for imp in pe.imports:
                dll_name = imp.name.lower()
                if dll_name not in benign_bank["imports_dict"]:
                    benign_bank["imports_dict"][dll_name] = set()
                
                for entry in imp.entries:
                    if entry.name: 
                        benign_bank["imports_dict"][dll_name].add(entry.name)

        except Exception as e:
            print(f"[!] Skipping {filename} due to parsing error: {e}")

    for dll in benign_bank["imports_dict"]:
        benign_bank["imports_dict"][dll] = list(benign_bank["imports_dict"][dll])

    benign_bank["code_bytes"] = [b for b in benign_bank["code_bytes"] if b]
    benign_bank["data_bytes"] = [b for b in benign_bank["data_bytes"] if b]
    benign_bank["resource_bytes"] = [b for b in benign_bank["resource_bytes"] if b]
    benign_bank["dos_stubs"] = [b for b in benign_bank["dos_stubs"] if b]

    with open(pickle_output_file, 'wb') as f:
        pickle.dump(benign_bank, f)

    print(f"\n[*] Extraction successfully completed for {valid_files} files.")
    print(f"[*] Artifacts saved to: {pickle_output_file}")
    print(f"[*] Extracted Component Counts:")
    print(f"    - Code Samples:     {len(benign_bank['code_bytes'])}")
    print(f"    - Data Samples:     {len(benign_bank['data_bytes'])}")
    print(f"    - Resource Samples: {len(benign_bank['resource_bytes'])}")
    print(f"    - DOS Stubs:        {len(benign_bank['dos_stubs'])}")
    print(f"    - Section Meta:     {len(benign_bank['section_meta'])}")
    print(f"    - Indexed DLLs:     {len(benign_bank['imports_dict'])}")

def main():
    parser = argparse.ArgumentParser(description="Extract benign PE components into a serialized bank.")
    
    parser.add_argument(
        "--benign_dir", 
        required=True, 
        help="Path to the directory containing benign .exe and .dll files"
    )
    parser.add_argument(
        "--output_dir", 
        default="./data/benign_content", 
        help="Path to the output directory for artifacts (default: ./data/benign_content)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.benign_dir):
        print(f"[-] Error: The directory '{args.benign_dir}' does not exist.")
        return

    build_benign_bank(args.benign_dir, args.output_dir)

if __name__ == "__main__":
    main()