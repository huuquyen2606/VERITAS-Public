# Windows Defender Batch Analysis Tool

This directory contains scripts for performing bulk, automated malware scanning using the standard **Windows Defender** command-line utility (`MpCmdRun.exe`) shipped with Windows 10. It is designed to evaluate evasion rates of adversarial malware against a standard, unmodified Windows Defender environment.

## Overview

The `defender_analysis.py` script systematically scans datasets of Portable Executable (PE) files. It manages progress by saving the scan state incrementally, which allows for resuming large scans without starting over.

### Key Features
- **Unmodified Environment**: Leverages the default Windows 10 Defender via `MpCmdRun.exe` without any custom modifications.
- **Safe Scanning**: Uses the `-DisableRemediation` flag to ensure that detected malware is not automatically quarantined or deleted, preserving your datasets.
- **Progress Tracking**: Scans are tracked in `defender_results.json`. The script will skip already processed files, making it safe to interrupt and resume.
- **Atomic Saving**: Progress is saved every 10 files using atomic file writes to prevent corruption during unexpected shutdowns.
- **Comprehensive Reporting**: Generates a final summary detailing total PE files scanned, number of detected/bypassed files, scan errors, and the overall evasion rate.

## Requirements

- **Operating System**: Windows 10
- **Windows Defender**: Must be enabled and `MpCmdRun.exe` must exist at `C:\Program Files\Windows Defender\MpCmdRun.exe`.

## Usage

1. **Configure Paths**: 
   Edit the script to update the `DATASETS_DIR` and `PROGRESS_FILE` variables to point to your local dataset folder and desired output JSON file.

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
2. **Run the Analysis**:
   ```bash
   python defender_analysis.py
   ```

3. **Output**:
   The script prints live progress to the console. Once finished, it outputs a summary table for each dataset subdirectory showing the evasion rate.

### How It Works
1. **Dataset Discovery**: The script looks inside `DATASETS_DIR` and treats each subdirectory as a separate dataset (e.g., `aimed`, `dqeaf`, `gamma`).
2. **PE File Validation**: It recursively searches for files and verifies if they are PE files by checking the extension (`.exe`, `.dll`, `.sys`, `.scr`) or the presence of the `MZ` header.
3. **Scanning**: Passes each file to `MpCmdRun.exe`. Return code `0` indicates the file bypassed detection, while return code `2` indicates a threat was detected.
4. **Resumption**: Before scanning, it compares the dataset against the `defender_results.json` file. Only pending, unscanned files are processed.
