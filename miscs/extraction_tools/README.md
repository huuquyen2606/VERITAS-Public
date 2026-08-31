# Extraction Tools

This directory contains scripts dedicated to extracting and fusing both static and dynamic features from malware datasets. The extracted data is ultimately saved into compressed NumPy archives (`.npz`) for downstream machine learning and heuristic evaluation.

## Files Overview

### 1. `static_features_extract.py`
A static analysis extraction script that processes PE (Portable Executable) files to extract code-level static properties.
- **Tools Used**: `pefile` for PE parsing, `capstone` for x86/x64 disassembly.
- **Features Extracted**:
  - **Raw Bytes**: Raw binary data of the file.
  - **Opcodes**: Disassembled mnemonic instructions from all executable sections (using Capstone).
  - **Static APIs**: Imported functions extracted from the PE's Import Directory.
- **Output**: Saves the extracted features, file names, and labels into a compressed `.npz` file (e.g., `malguise_static.npz`).

### 2. `dynamic_features_extract.py`
An ultimate dynamic extraction pipeline built for interacting with a cluster of CAPEv2 sandbox servers.
- **Cluster Integration**: Distributes samples across multiple CAPE servers via API and SSH.
- **Integrity Checks**: Validates analysis integrity by ensuring minimum execution duration and absence of specific sandbox evasion signatures (e.g., Aimed-style integrity errors).
- **Features Extracted**:
  - **Behavioral APIs**: Monitored API calls during execution.
  - **PE Information**: Sections, imports, and CAPE signatures.
  - **Syscall Traces**: Extracted system call traces from sandbox behavior reports.
- **Workflow**: Submits samples in batches, processes reports continuously, and saves partial results in `.npz` batch files with automatic merging upon completion.

### 3. `group_dynamic_static.py`
A data fusion script that merges previously extracted static and dynamic `.npz` files into a unified representation.
- **Memory Management**: Uses `psutil` to dynamically check available RAM. Automatically switches between a "Fast Mode" (loading all properties into memory) and a "Safe Mode" (lazy loading/mapping) to prevent OOM errors on large datasets.
- **Alignment**: Cross-references samples with the physical dataset directory to guarantee that only samples with both valid static and dynamic traces are included.
- **Output**: Generates a consolidated feature dataset (`extracted` directory) containing labels, imports, sections, APIs (both Cuckoo and PE-based), opcodes, and raw bytes.

## Usage
1. Configure dataset paths (e.g., `TARGET_DIRECTORY` or `DATASET_ROOT`) directly within the scripts.
2. For dynamic extraction, configure your CAPEv2 cluster IP addresses and credentials inside `dynamic_features_extract.py`.
3. Run the static and dynamic extractors separately.
4. Finally, execute `group_dynamic_static.py` to merge the results into a unified dataset.
