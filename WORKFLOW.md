# Adversarial Malware Evaluation Workflow

## Overview
This repository contains a comprehensive suite for evaluating **VERITAS**, our proposed adversarial malware technique. It covers the entire pipeline from data extraction and baseline detection to adversarial sample generation, functionality verification, and final metric calculation.

**Important Context:** The primary purpose of this repository is to showcase and evaluate our proposed technique (`proposed/`). All other adversarial techniques (`baseline_adversarial_techniques/`) and detectors (`baseline_detectors/`) are included strictly as **comparators** to establish a baseline. The miscellaneous tools (`miscs/`) and evaluating codes are helper utilities designed to set up the robust evaluation workflow.

This document serves as a high-level conceptual reference for the data flow and architectural components. For specific implementation details and execution commands, please consult the `README.md` files located within each respective directory.

---

## Directory Structure and Component Reference

*   **`proposed/` (Core Contribution)**: Contains the implementation of VERITAS, our newly proposed adversarial malware technique. 
*   **`baseline_adversarial_techniques/` (Comparators)**: Implementations of existing evasion attacks (e.g., GAMMA, MalGPT, DQEAF, AIMED-RL) designed to act as baselines against our proposed method.
*   **`integrity_functionaltiy_evaluating_codes/`**: A robust evaluation suite that verifies whether the adversarial modifications broke the malware's original functionality using static and dynamic trace alignment.
*   **`detector_evaluaion/`**: Scripts responsible for aggregating the final evasion and functionality metrics.
*   **`our_results/`**: The designated output directory for final evaluation reports and metrics.

---

## Data Management & File Placement

The workflow relies heavily on structured datasets and intermediate artifacts:
1.  **Raw Malware (PE Files)**: Expected to be organized into distinct subdirectories by technique (e.g., `datasets/gamma/`, `datasets/malgpt/`) when feeding into heuristic detectors or adversarial generators.
2.  **Feature Archives (`.npz`)**: Both the ML detectors and functionality evaluators rely on `.npz` files (e.g., `datasets/APIs/`, `datasets/syscalls/`). These contain the extracted features and behavioral traces.
3.  **Intermediate Reports (`.json`)**: Scanning tools and evaluation scripts generate detailed JSON reports (e.g., `defender_results.json`, `dqeaf_detailed_reports.json`) which are used by downstream filtering scripts to maintain state and accuracy.

---

## The End-to-End Workflow Pipeline

The methodology follows a strict five-phase pipeline. Researchers should follow this logical flow when running experiments.

### Phase 1: Feature Extraction & Sandbox Setup (`miscs/`)
The process begins by preparing the environment and establishing the ground truth.
*   **Environment Setup**: The CAPEv2 sandbox is configured with custom BinSim/Stalker instrumentation to ensure deep, evasion-resistant execution tracing.
*   **Data Extraction**: Original malware samples undergo static and dynamic analysis. The results are fused into `.npz` datasets, serving as the baseline for both ML training and functionality verification.

### Phase 2: Baseline Detection Evaluation (`baseline_detectors/`)
Before any evasion techniques are applied, the original dataset is evaluated to establish baseline detection rates.
*   **Machine Learning**: Models (e.g., MalConv) are trained and tested on the extracted `.npz` datasets.
*   **Heuristics**: Raw PE files are bulk-scanned using ClamAV, Windows Defender, and VirusTotal to record their initial detected/bypassed status.

### Phase 3: Adversarial Sample Generation (`proposed/` & `baseline_adversarial_techniques/`)
Adversarial techniques are employed to mutate the original malware samples into evasive variants.
*   **VERITAS (Proposed)**: Our novel approach is executed to generate the primary set of evasive samples for evaluation.
*   **Baselines (Comparators)**: Techniques like GAMMA, AIMED-RL, and MalGPT are executed to generate their respective datasets of adversarial PE files.
*   This phase produces distinct datasets of *adversarial PE files* (one for our proposed method and one for each baseline) that will be pitted against each other in the final evaluation.

### Phase 4: Integrity & Functionality Verification (`integrity_functionaltiy_evaluating_codes/`)
A critical, scientifically rigorous step: ensuring the evasive modifications did not destroy the malware's actual malicious behavior.
*   **Integrity Checks**: Verifying the new adversarial samples can execute without crashing.
*   **Semantic Verification**: Using Control Flow Graphs (CFG), Symbolic Execution (Z3), and sequence alignment (Smith-Waterman) to compare the adversarial trace against the original Phase 1 trace.
*   **Dataset Filtering**: Samples that fail functionality checks are permanently stripped out. Only structurally sound, functional malware proceeds to the final evaluation.

### Phase 5: Final Result Evaluation (`detector_evaluaion/` & `our_results/`)
The surviving, fully functional adversarial samples undergo a final assessment.
*   The functional dataset is re-scanned by the baseline ML and heuristic detectors.
*   Scripts in `detector_evaluaion/` synthesize the data to calculate the ultimate metrics: combining Functionality & Integrity Rates (FIR) with the final Evasion/Bypass Rates.
*   Final aggregated reports are saved in `our_results/`.

---
*For specific execution instructions, environment setups, and configuration options for any component, please navigate to its respective directory and consult the local `README.md`.*
