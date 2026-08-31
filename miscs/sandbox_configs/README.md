# Sandbox Configurations

This directory contains custom execution packages and instrumentation scripts designed for the CAPEv2 sandbox. These scripts are responsible for advanced dynamic analysis, trace collection, and mitigating malware evasion techniques.

## Files Overview

### 1. `binsim.py`
A customized CAPEv2 Execution Package tailored for the "BinSim" tracing methodology.
- **Smart Architecture Handling**: Dynamically identifies the target PE's architecture (32-bit vs. 64-bit) and type (EXE vs. DLL). Effectively uses `Sysnative` and `SysWOW64` for proper 64-bit environment emulation, particularly for running DLLs via `rundll32.exe`.
- **Frida Server Bridge**: Instead of native CAPE injection, it connects to a 64-bit `frida-server` running in the guest VM. This bypasses common anti-debugging and architecture-mismatch issues.
- **Trace Management**: Intercepts tracing messages sent from the guest JavaScript, writing them reliably to disk in chunks to avoid infinite loops and buffer constraints. Uploads the finalized JSON trace back to the CAPE host.

### 2. `binsim_stalker.js`
The core Frida instrumentation payload injected by `binsim.py` into the target malware process.
- **Strict Syscall Mapping**: Hooks a critical whitelist of Windows APIs and NT Syscalls (e.g., `NtCreateFile`, `NtAllocateVirtualMemory`, `NtCreateUserProcess`) that strictly match Triton backend standards and academic dataset requirements.
- **GUI-Safe JIT Stalker**: Implements Frida's `Stalker` to perform block-level dynamic instruction tracing. Captures assembly instructions executed just before critical API invocations, providing deep context on the malware's behavior.
- **Module Filtering**: Differentiates between actual malware code and legitimate Windows API calls (e.g., filtering out `C:\Windows\*`), ensuring that the resulting trace buffer only contains malicious instruction sequences.

### 3. `conf/`
Directory intended for CAPEv2 sandbox specific configuration files that tune the analysis environment and analysis routing.

## Usage
- The `binsim.py` file should be placed inside CAPEv2's `analyzer/windows/modules/packages/` directory (or used as a custom package).
- The `binsim_stalker.js` file will be automatically copied and injected by the `binsim.py` script during sandbox execution. Ensure the path references in `binsim.py` correctly point to where `binsim_stalker.js` is stored on the host.
