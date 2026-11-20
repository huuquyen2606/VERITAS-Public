# VERITAS

**VERITAS: Validity-Constrained Multi-Surface Adversarial Malware Generation**

Anonymous Authors

VERITAS is a validity-constrained framework that couples broad transformation coverage with search-time operational-validity enforcement for problem-space adversarial malware generation. It exposes 19 realizable actions spanning PE structure and metadata, code and control flow, and runtime-observable API behavior.

This repository accompanies the research paper and is intended to support research reproducibility.

## Paper Highlights

- VERITAS presents a validity-constrained framework that couples broad transformation coverage with search-time operational-validity enforcement.
- It introduces a unified 19-action transformation space spanning PE structure and metadata, code/CFG, and runtime-observable behavior.
- Search-time operational-validity enforcement uses a fail-closed validity-gated candidate admission mechanism assessing execution viability and behavioral preservation.
- Evaluated across nine held-out learning-based detectors, VERITAS achieves a mean validity-conditioned attack success rate of 23.13%, reaching 73.05% on RTF-BERT.
- Variants optimized against a single local surrogate transfer to unseen operational targets, achieving 23.16% Valid-ASR against ClamAV and 14.89% against Microsoft Defender.

## Paper Overview

The paper formulates problem-space adversarial malware generation as malware-to-benign evasion subject to explicit operational-validity requirements. During sequential optimization, each tentative candidate is evaluated for execution viability and behavioral preservation using complementary API and low-level system-call evidence.

VERITAS exposes 19 realizable actions across three complementary surfaces:

- **PE structure and metadata (11 actions)**: headers, DOS stub, sections, imports, overlays, certificate/debug directories, checksums, and timestamps.
- **Code, control flow, and whole-binary transformations (7 actions)**: instruction rewriting, code randomization, lift-transform-recompile artifacts (RetDec + OLLVM), call-edge redividing, semantic-NOP insertion, external packing (UPX), and encryption loaders (Darkarmour).
- **Runtime-observable behavior (1 action)**: observable API footprint modification through lightweight trampoline redirection logic.

The framework separates executable evolution from policy-state admission, ensuring that only validity-qualified candidate representations influence the adversarial policy while retaining validity failures as learning feedback.

## Repository Structure

```text
.
|-- README.md
|-- proposed/
|-- baseline_adversarial_techniques/
|-- baseline_detectors/
|-- integrity_functionaltiy_evaluating_codes/
|-- detector_evaluaion/
|-- our_results/
`-- miscs/
```

- **`proposed/`**: Contains the implementation of VERITAS, our newly proposed adversarial malware technique.
- **`baseline_adversarial_techniques/`**: Implementations of existing evasion attacks (e.g., GAMMA, MalGPT, DQEAF, AIMED-RL) designed to act as baselines.
- **`baseline_detectors/`**: Scripts and code for baseline detectors (e.g., MalConv, ClamAV, Windows Defender).
- **`integrity_functionaltiy_evaluating_codes/`**: A robust evaluation suite that verifies whether the adversarial modifications broke the malware's original functionality using static and dynamic trace alignment.
- **`detector_evaluaion/`**: Scripts responsible for aggregating the final evasion and functionality metrics.
- **`our_results/`**: The designated output directory for final evaluation reports and metrics.
- **`miscs/`**: Helper utilities designed to set up the robust evaluation workflow.

## Data Management & File Placement

The workflow relies heavily on structured datasets and intermediate artifacts:

1. **Raw Malware (PE Files)**: Expected to be organized into distinct subdirectories by technique (e.g., `datasets/gamma/`, `datasets/malgpt/`) when feeding into heuristic detectors or adversarial generators.
2. **Feature Archives (`.npz`)**: Both the ML detectors and functionality evaluators rely on `.npz` files (e.g., `datasets/APIs/`, `datasets/syscalls/`). These contain the extracted features and behavioral traces.
3. **Intermediate Reports (`.json`)**: Scanning tools and evaluation scripts generate detailed JSON reports (e.g., `defender_results.json`, `dqeaf_detailed_reports.json`) which are used by downstream filtering scripts to maintain state and accuracy.

## The End-to-End Workflow Pipeline

The methodology follows a strict five-phase pipeline. Researchers should follow this logical flow when running experiments.

### Phase 1: Feature Extraction & Sandbox Setup (`miscs/`)

The process begins by preparing the environment and establishing the ground truth.

- **Environment Setup**: The CAPEv2 sandbox is configured with custom BinSim/Stalker instrumentation to ensure deep, evasion-resistant execution tracing.
- **Data Extraction**: Original malware samples undergo static and dynamic analysis. The results are fused into `.npz` datasets, serving as the baseline for both ML training and functionality verification.

### Phase 2: Baseline Detection Evaluation (`baseline_detectors/`)

Before any evasion techniques are applied, the original dataset is evaluated to establish baseline detection rates.

- **Machine Learning**: Models (e.g., MalConv) are trained and tested on the extracted `.npz` datasets.
- **Heuristics**: Raw PE files are bulk-scanned using ClamAV, Windows Defender, and VirusTotal to record their initial detected/bypassed status.

### Phase 3: Adversarial Sample Generation (`proposed/` & `baseline_adversarial_techniques/`)

Adversarial techniques are employed to mutate the original malware samples into evasive variants.

- **VERITAS (Proposed)**: Our novel approach is executed to generate the primary set of evasive samples for evaluation.
- **Baselines (Comparators)**: Techniques like GAMMA, AIMED-RL, and MalGPT are executed to generate their respective datasets of adversarial PE files.
- This phase produces distinct datasets of _adversarial PE files_ (one for our proposed method and one for each baseline) that will be pitted against each other in the final evaluation.

### Phase 4: Integrity & Functionality Verification (`integrity_functionaltiy_evaluating_codes/`)

A critical, scientifically rigorous step: ensuring the evasive modifications did not destroy the malware's actual malicious behavior.

- **Integrity Checks**: Verifying the new adversarial samples can execute without crashing.
- **Semantic Verification**: Using Control Flow Graphs (CFG), Symbolic Execution (Z3), and sequence alignment (Smith-Waterman) to compare the adversarial trace against the original Phase 1 trace.
- **Dataset Filtering**: Samples that fail functionality checks are permanently stripped out. Only structurally sound, functional malware proceeds to the final evaluation.

### Phase 5: Final Result Evaluation (`detector_evaluaion/` & `our_results/`)

The surviving, fully functional adversarial samples undergo a final assessment.

- The functional dataset is re-scanned by the baseline ML and heuristic detectors.
- Scripts in `detector_evaluaion/` synthesize the data to calculate the ultimate metrics: combining Functionality & Integrity Rates (FIR) with the final Evasion/Bypass Rates.
- Final aggregated reports are saved in `our_results/`.

## Note on this Repository

This repository provides the complete modular implementation of the VERITAS framework (`proposed/`), the reproduced problem-space adversarial baselines (`baseline_adversarial_techniques/`), the nine learning-based and three operational target detector integrations (`baseline_detectors/`), and the multi-validator operational preservation evaluation suite (`integrity_functionaltiy_evaluating_codes/` and `detector_evaluaion/`).

The primary purpose of this repository is to showcase and evaluate our proposed technique (`proposed/`). All other adversarial techniques (`baseline_adversarial_techniques/`) and detectors (`baseline_detectors/`) are included as comparators to support reproducible benchmark evaluation.

For specific execution instructions, environment setups, and configuration options for any component, please navigate to its respective directory and consult the local `README.md`.

## Main Results Summary

- **Generation Coverage & Operational Validity**: Finalizes valid variants for 1,203 of 1,794 holdout malware samples (**67.06% Generation Coverage**) with **100.00%** joint operational validity ($V_{\mathrm{VER}}$).
- **Cross-Detector Attack Effectiveness**: Achieves an unweighted mean validity-conditioned attack success rate ($\mathrm{V\mbox{-}ASR}^{\mathrm{VER}}$) of **23.13%** across nine held-out learning-based detectors (leading on five targets and reaching **73.05%** on RTF-BERT, **37.17%** on multiclass M-Attn-Health, **34.47%** on binary M-Attn-Health, **15.37%** on multiclass MalConv, and **13.90%** on MultiView CNN).
- **Operational Transferability**: Variants optimized against a single local surrogate transfer without target feedback, achieving **23.16%** $\mathrm{V\mbox{-}ASR}^{\mathrm{VER}}$ against ClamAV and **14.89%** against Microsoft Defender (**19.03%** standalone engine mean) and a **16.42%** mean across VirusTotal TER thresholds (10%–80%).
- **Low Transformation Burden**: Averages **1.39 transformation steps** (maximum 5) and **51.89% executable-size growth**, compared with 321.21% for GAMMA and 901.52% for OBFU-mal.

## Authors / Maintainers

- Anonymous Authors

## License

This repository is released under the Apache License 2.0. See [LICENSE](LICENSE) for details.

