# ClamAV Bulk Malware Analysis Tool

This directory contains the tools necessary to evaluate the evasion rates of adversarial malware datasets against **ClamAV**, an open-source antivirus engine. It includes a custom configuration optimized for bulk, deep scanning and a Python automation script.

## Overview

The `clamav_analysis.py` script leverages the `pyclamd` library to interact with a running ClamAV daemon (`clamd`). This approach avoids the overhead of spawning a new process for each scan, making it significantly faster for bulk analysis.

### Key Features
- **Daemon-based Scanning**: Connects via TCP to a local ClamAV daemon (`127.0.0.1:3310`), allowing high-throughput scanning.
- **Custom Configuration (`clamd.conf`)**: An included config file optimized for bulk malware analysis. It features increased thread limits, extended timeouts, and increased depth limits for unpacking and de-obfuscation (e.g., max scan time, max recursion, increased PCRE limits).
- **Resilient Progress Tracking**: Scan results are saved to `clamav_results.json`. The script will resume from where it left off if interrupted.
- **Atomic Operations**: State is saved every 10 files using an atomic `.tmp` file replacement to prevent data corruption.
- **Automatic PE Verification**: Identifies executables by checking standard extensions (`.exe`, `.dll`, etc.) and reading the file header for the `MZ` magic bytes.

## Requirements

- **Python Packages**: `pyclamd`
- **ClamAV**: Must be installed and the `clamd` service must be running.

## Usage

### 1. Configure ClamAV
Apply the provided `clamd.conf` configuration to your ClamAV installation. This config enables the TCP socket on port 3310 and configures limits for deeper scanning.

*Start the ClamAV daemon:*
```bash
net start clamd
```
*(Or the equivalent command for your operating system)*

### 2. Configure the Script
Edit `clamav_analysis.py` to point `DATASETS_DIR` and `PROGRESS_FILE` to your dataset location and desired output file path.

**Expected Dataset Structure:**
The `DATASETS_DIR` must contain subdirectories where each subdirectory represents a distinct dataset (e.g., different evasion techniques). The script will recursively scan all PE files inside these subdirectories.
```text
DATASETS_DIR/
├── aimed/
│   ├── malware1.exe
│   └── malware2.dll
├── gamma/
│   ├── nested_folder/
│   │   └── malware3.exe
│   └── malware4.exe
└── original_malware/
    └── ...
```
### 3. Run the Analysis
```bash
python clamav_analysis.py
```

## How It Works
1. **Connection Validation**: The script attempts to ping the ClamAV daemon via the local network socket. It will fail early if the service is unreachable.
2. **Dataset Processing**: It iterates over each subdirectory in `DATASETS_DIR`, treating them as separate test sets.
3. **Execution**: Unscanned PE files are sent to the ClamAV daemon. The daemon returns either `None` (bypassed) or a tuple containing the detected malware signature.
4. **Summary**: Upon completion, a summary report is printed displaying the total scanned files, detected threats, bypassed files, and the calculated evasion percentage for each dataset.
