# VERITAS

**VERITAS: Validity-Constrained Multi-Surface Adversarial Malware Generation**

*Anonymous Authors*


VERITAS is a validity-constrained reinforcement learning framework for generating problem-space adversarial Windows Portable Executable (PE) malware. It couples broad multi-surface transformation coverage with search-time operational-validity enforcement, ensuring generated adversarial variants evade target detectors while strictly preserving functional and behavioral integrity.

This repository accompanies the research paper and is intended to support research reproducibility.

---

## Paper Highlights

- Introduces a multi-surface search space over 19 realizable PE perturbation operators spanning headers, section layouts, instruction sequences, control-flow graph (CFG) topology, and binary lifting via Obfuscator-LLVM (OLLVM).
- Implements a fail-closed state-admission mechanism that evaluates tentative candidates for execution viability and semantic preservation before exposing representations to the agent policy.
- Employs a multi-branch Distributional Dueling DQN (C51) with Prioritized Experience Replay (PER) and Factorized NoisyNet exploration to robustly model high-variance sandbox feedback across 51 discrete atoms over the support `[-500, 200]`.
- Implements a triple-layer semantic validity verification engine combining structural gating, formal SMT symbolic verification (Triton x86 lifter + Z3 SMT-LIB2 solver), and accelerated C++ Smith-Waterman sequence alignment with affine gap penalty and difference pruning (`td = 0.02`).
- Achieves high validity-conditioned attack success rates across heterogeneous learning-based detectors (up to 73.05% on RTF-BERT) and transfers effectively to commercial engines, achieving 23.16% Valid-ASR against ClamAV and 14.89% against Microsoft Defender.
- Utilizes an asynchronous microservice architecture backed by Redis to decouple heavy binary analysis and sandbox execution from the RL optimization loop.

---

## Paper Overview

Problem-space adversarial attacks against machine learning-based malware detectors must generate realizable binaries whose operational properties remain consistent with the original malware. VERITAS addresses this challenge by integrating formal and dynamic validity constraints directly into sequential optimization.

The framework operates across the following core components:

- **19-Action Multi-Surface Perturbation Engine**: Executes non-destructive header manipulation, section slack shifting, in-place NOP sled insertion, CFG edge redivision, basic block randomization, and full OLLVM decompilation-recompilation.
- **2,717-Dimensional Multi-Modal State Builder**: Unifies static features (EMBER v2 2,381-dim), dynamic API execution sequences (184-dim), CFG graph structural topology (112-dim), and RL environment metadata (40-dim).
- **Triple-Layer Semantic Verification Engine**:
  - *Layer 1 (Integrity Filter)*: Fast structural and dynamic viability gating.
  - *Layer 2 (BinSim SMT Prover)*: Triton-based instruction lifting and Z3 formal equivalence checking over critical syscall slices.
  - *Layer 3 (C++ Sequence Aligner)*: C++ Smith-Waterman alignment with sensitive API anchoring (`wb = 20`) and difference pruning (`td = 0.02`).
- **Asynchronous Worker Infrastructure**: Redis-backed queues dispatching parallel feature extraction tasks across dedicated workers (`API`, `Syscall`, `Angr`, `Static`).

The framework is evaluated on 4,785 PE malware and benign samples across five prominent malware families:

- Locker (Ransomware)
- Mediyes (Trojan / Adware)
- Winwebsec (Rogue Security Software)
- Zbot (Banking Trojan)
- Zeroaccess (Rootkit / Botnet)

---

## Repository Structure

```text
.
|-- README.md                               # Main framework documentation
|-- requirements.txt                        # Python dependencies
|-- actions/                                # Realizable PE mutation operators
|   |-- base_action.py                      # Base interface for mutation actions
|   |-- bytecode_api_hijacking_action.py    # Action 16: Dynamic trampoline call indirection
|   |-- cfg_nop_actions.py                  # Actions 17 & 18: CFG edge redivision and NOP injection
|   |-- darkarmour_action.py                # Action 14: DarkArmour XOR payload encryption
|   |-- packer_action.py                    # Action 12: UPX packing transformation
|   `-- pe_mutator.py                       # PEMutator dispatching all 19 mutation actions
|-- agent/                                  # Distributional RL agent
|   |-- agent_network.py                    # Multi-branch Distributional Dueling network (51 Atoms)
|   |-- categorical_projection.py           # C51 categorical distribution projection
|   |-- distributional_loss.py              # Cross-entropy loss for categorical distributions
|   |-- per_buffer.py                       # Prioritized Experience Replay buffer (SumTree)
|   `-- veritas_agent.py                    # VERITASAgent training and inference orchestrator
|-- configs/                                # Experiment YAML configurations
|   |-- train.yaml                          # Training hyperparameters and dataset paths
|   `-- test.yaml                           # Evaluation settings and checkpoint paths
|-- env/                                    # Environment and microservice bridges
|   |-- adv_rl_bridge.py                    # Redis-backed bridge for worker orchestration
|   |-- malware_env.py                      # Gymnasium MalwareEnv implementation
|   `-- adv_RL_env/                         # Backend evaluation microservices
|       |-- core/
|       |   `-- redis_orchestrator.py       # CAPE round-robin and barrier synchronization
|       |-- evaluation/
|       |   |-- arch.py                     # Multi-attention surrogate detector architecture
|       |   |-- detector_api.py             # Multimodal detector evaluation pipeline
|       |   |-- functionality_evaluator.py  # BinSim (Triton+Z3) and Smith-Waterman engine
|       |   |-- reward_calculator.py        # Parameterized objective reward calculator
|       |   `-- sequence_deduplicator.py    # Dynamic trace loop compressor
|       |-- examples/comparing_apis/        # C++ Smith-Waterman Pybind11 extension
|       |   |-- smith-wanderman.cpp         # C++ local sequence alignment with affine gaps
|       |   `-- setup.py                    # Pybind11 build script
|       `-- workers/                        # Background feature extraction daemons
|           |-- worker_angr.py              # CFGFast analysis worker
|           |-- worker_api.py               # CAPEv2 API trace extraction worker
|           |-- worker_static.py            # Raw byte stream worker
|           `-- worker_syscall.py           # Syscall trace and memory slice worker
|-- features/
|   `-- state_builder.py                    # 2,717-dim StateBuilder (EMBER, API, CFG, Env)
`-- scripts/                                # CLI entrypoints and automation tools
    |-- train.py                            # End-to-end agent training
    |-- test.py                             # Evaluation and adversarial example generation
    |-- rl_entrypoint_utils.py              # Shared checkpointing and logging utilities
    |-- create_eval_shards.py               # Dataset sharding tool for parallel evaluation
    |-- merge_eval_shards.py                # Consolidated shard metric aggregator
    |-- prepare_benign_content.py           # Benign content bank harvesting tool
    |-- precompute_code_randomize.py        # Offline analysis for Action 15 (angr/Capstone/Keystone)
    |-- precompute_code_translation.py      # Offline RetDec + OLLVM pipeline for Action 3
    |-- start_eval_workers.sh               # Worker startup script
    |-- stop_eval_workers.sh                # Worker shutdown script
    `-- start_functionality_evaluator.sh    # Functionality evaluator service launcher
```

---

## Installation

### 1. Prerequisites and System Packages

VERITAS requires a 64-bit Linux environment (Ubuntu 20.04/22.04 LTS recommended) with Python 3.10+, Redis, and standard compilation toolchains.

Install required system packages:

```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    redis-server \
    clang-14 \
    llvm-14 \
    llvm-14-tools \
    binutils-mingw-w64 \
    gcc-mingw-w64 \
    g++-mingw-w64 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tmux

# Start Redis service
sudo systemctl start redis-server
```

### 2. Python Environment Setup

Create and activate the main virtual environment for the agent:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 3. Backend Functionality Environment (`func_venv`)

The Functionality Evaluator and CFG worker (angr) require a dedicated virtual environment with Python <= 3.12. Create it inside the `env/adv_RL_env` directory:

```bash
python3 -m venv env/adv_RL_env/func_venv
env/adv_RL_env/func_venv/bin/python -m pip install --upgrade pip setuptools wheel
env/adv_RL_env/func_venv/bin/python -m pip install -r env/adv_RL_env/requirements_functionality.txt
```

### 4. Build the C++ Smith-Waterman Extension (`fast_sw`)

Compile and install the accelerated C++ sequence alignment extension into BOTH environments:

```bash
# Install for main agent environment
pip install ./env/adv_RL_env/examples/comparing_apis

# Install for backend functionality environment
env/adv_RL_env/func_venv/bin/python -m pip install ./env/adv_RL_env/examples/comparing_apis
```

### 5. Verification Check

Verify that all dependencies and compiled modules are correctly loaded:

```bash
python3 -c "import torch, lief, angr, z3, triton, redis, fast_sw; print('All core dependencies imported successfully.')"
env/adv_RL_env/func_venv/bin/python -c "import angr, triton, z3, fast_sw; print('Backend dependencies imported successfully.')"
```

---

## Dataset Preparation

### 1. Dataset Layout

The dataset is partitioned into training (`Adv_agent/`) and holdout evaluation (`Test/`) splits across five malware families and a benign reference set:

```text
data/dataset/
|-- Adv_agent/                  # Training split (2,991 total samples)
|   |-- Benign/         (300)   # Clean binaries for benign bank harvesting
|   |-- Locker/          (99)   # Ransomware
|   |-- Mediyes/        (435)   # Trojan / Adware
|   |-- Winwebsec/     (1320)   # Rogue Security Software
|   |-- Zbot/           (630)   # Banking Trojan
|   `-- Zeroaccess/     (207)   # Rootkit / Botnet
`-- Test/                       # Evaluation split (1,994 total samples)
    |-- Benign/         (200)   # Clean evaluation `binaries
    |-- Locker/          (66)
    |-- Mediyes/        (290)
    |-- Winwebsec/      (880)
    |-- Zbot/           (420)
    `-- Zeroaccess/     (138)
```

### 2. Step 1: Harvest the Benign Content Bank

Extract benign byte sequences, imports, DOS stubs, and section headers from clean binaries:

```bash
python3 scripts/prepare_benign_content.py \
    --benign_dir data/dataset/Adv_agent/Benign \
    --output_dir data/benign_content
```

Verify output artifact:

```bash
ls -lh data/benign_content/benign_bank.pkl
```

### 3. Step 2: Offline Precomputation (Actions 3 and 15)

To eliminate runtime latency during RL rollouts, computationally heavy transformations are precomputed offline:

#### A. Code Randomization (`CODE_RANDOMIZE` - Action 15)

```bash
# Precompute Training Set
for family in Locker Mediyes Winwebsec Zbot Zeroaccess; do
    python3 scripts/precompute_code_randomize.py \
        --input-dir data/dataset/Adv_agent/$family \
        --output-dir data/precomputed/train/code-randomize \
        --seed 1337 --max-rewrites 200 --timeout-sec 180
done

# Precompute Test Set
for family in Locker Mediyes Winwebsec Zbot Zeroaccess; do
    python3 scripts/precompute_code_randomize.py \
        --input-dir data/dataset/Test/$family \
        --output-dir data/precomputed/test/code-randomize \
        --seed 1337 --max-rewrites 200 --timeout-sec 180
done
```

#### B. Code Translation via OLLVM (`CODE_TRANSLATION` - Action 3)

```bash
RETDEC=utils/RetDec/bin/retdec-decompiler
OLLVM=utils/obfuscator_llvm_14/build/bin

# Precompute Training Set
for family in Locker Mediyes Winwebsec Zbot Zeroaccess; do
    python3 scripts/precompute_code_translation.py \
        --input-dir data/dataset/Adv_agent/$family \
        --output-dir data/precomputed/train/code-translation \
        --work-dir /dev/shm/ct_work \
        --retdec-bin $RETDEC \
        --clang-bin /usr/bin/clang-14 \
        --opt-bin $OLLVM/opt \
        --dlltool-bin /usr/bin/llvm-dlltool-14 \
        --llvm-dis-bin /usr/bin/llvm-dis-14 \
        --use-opt \
        --obf-flags "-passes=obf-sub,obf-fla,obf-bcf -sub -fla -bcf" \
        --timeout-sec 600
done

# Precompute Test Set
for family in Locker Mediyes Winwebsec Zbot Zeroaccess; do
    python3 scripts/precompute_code_translation.py \
        --input-dir data/dataset/Test/$family \
        --output-dir data/precomputed/test/code-translation \
        --work-dir /dev/shm/ct_work \
        --retdec-bin $RETDEC \
        --clang-bin /usr/bin/clang-14 \
        --opt-bin $OLLVM/opt \
        --dlltool-bin /usr/bin/llvm-dlltool-14 \
        --llvm-dis-bin /usr/bin/llvm-dis-14 \
        --use-opt \
        --obf-flags "-passes=obf-sub,obf-fla,obf-bcf -sub -fla -bcf" \
        --timeout-sec 600
done
```

---

## Configuration

Experiment parameters are configured via YAML files in `configs/train.yaml` and `configs/test.yaml`.

Key configuration parameters include:

- `env.max_steps`: Maximum episode horizon ($T = 5$).
- `env.function_threshold`: Semantic functionality threshold ($T_F = 80.0\%$).
- `agent.num_actions`: Size of action space ($|\mathcal{A}| = 19$).
- `agent.num_atoms`: Distributional categorical atoms ($N = 51$).
- `agent.v_min` / `agent.v_max`: Support bounds of the value distribution (`[-500.0, 200.0]`).
- `agent.sigma_init`: Initial noise standard deviation for NoisyNet layers (`0.5`).
- `agent.gamma`: Discount factor ($\gamma = 0.99$).
- `agent.lr`: Adam learning rate ($1\times 10^{-4}$).
- `agent.per.capacity`: Prioritized Experience Replay buffer capacity (`10000`).
- `agent.per.alpha`: Priority exponent ($\alpha = 0.6$).
- `agent.per.beta`: Initial importance-sampling exponent ($\beta = 0.4$).
- `train.batch_size`: Mini-batch size (`32`).
- `train.warmup_steps`: Warmup transitions before learning starts (`500`).

---

## Running Experiments

### 1. Configure CAPE Sandbox Nodes
Before running, you must provide your own CAPEv2 Sandbox node IPs, SSH credentials, and API tokens for the evaluation workers.
Edit `assets/cape_nodes.json` (and its shard variants in `assets/` for parallel evaluation) to replace the placeholder `<YOUR_CAPE_NODE_X_IP>` and `<YOUR_SSH_PASSWORD>` with your actual setup.
Also, ensure you replace `<YOUR_CAPE_API_TOKEN>` in `env/adv_RL_env/malware_rl_system/workers/worker_api.py` and `worker_syscall.py`.

### 2. Launch Backend Microservices

Before training or evaluation, start the Redis broker, background feature extraction workers, and functionality evaluator daemon:

```bash
# 1. Start Redis
redis-server --daemonize yes
redis-cli ping   # Must return PONG

# 2. Start the 4 background feature extraction workers
cd env/adv_RL_env && bash start_workers.sh && cd ../..

# 3. Start the Functionality Evaluator listener daemon
bash scripts/start_functionality_evaluator.sh
```

### 2. Training the VERITAS Agent

Run end-to-end agent training using `scripts/train.py`:

```bash
python3 scripts/train.py \
    --config configs/train.yaml \
    --device cpu \
    --evaluate-reset \
    --prepare-cfg-calls \
    --inline-functionality \
    2>&1 | tee logs/train_full.log
```

Key CLI options:
- `--config`: Path to training YAML configuration file.
- `--episodes`: Total training episodes (defaults to dataset size).
- `--device`: Target compute device (`cuda` or `cpu`).
- `--limit`: Limits the number of episodes for smoke testing.
- `--resume`: Checkpoint path (`.pt`) to resume training from.
- `--inline-functionality`: Evaluates functionality within the main loop without requiring an external listener.

Monitor training metrics in real time:

```bash
tail -f logs/runs/run_002/train_history.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    print(f'ep={d[\"episode\"]} reward={d[\"total_reward\"]:.1f} evaded={d[\"evaded\"]} ER={d[\"evasion_rate\"]:.3f}')
"
```

### 3. Single-Process Evaluation

Evaluate a trained agent checkpoint on the holdout `Test/` dataset:

```bash
python3 scripts/test.py \
    --config configs/test.yaml \
    --checkpoint models/checkpoints/run_002/final.pt \
    --output-dir outputs/aes_test_run_002 \
    --evaluate-reset \
    --prepare-cfg-calls \
    --inline-functionality \
    2>&1 | tee logs/test_eval.log
```

View summary evaluation metrics:

```bash
python3 -m json.tool outputs/aes_test_run_002/summary_metrics.json
```

### 4. Parallel Multi-Shard Evaluation

For accelerated evaluation across multi-core servers, partition the holdout dataset into independent shards:

```bash
# 1. Generate balanced shard configurations
python3 scripts/create_eval_shards.py \
    --input-root data/dataset/Test \
    --output-root data/dataset/Test_shards/run_002 \
    --families Locker Mediyes Winwebsec Zbot Zeroaccess \
    --shards 3 \
    --mode symlink \
    --manifest outputs/aes_test_run_002_parallel/shard_manifest.json

# 2. Flush and launch worker pools on separate Redis DBs
redis-cli -n 1 FLUSHDB
redis-cli -n 2 FLUSHDB
redis-cli -n 3 FLUSHDB

bash scripts/start_eval_workers.sh --name shard_1 --redis-db 1 --work-dir /tmp/veritas_shard_1 --log-dir outputs/aes_test_run_002_parallel/logs/workers/shard_1
bash scripts/start_eval_workers.sh --name shard_2 --redis-db 2 --work-dir /tmp/veritas_shard_2 --log-dir outputs/aes_test_run_002_parallel/logs/workers/shard_2
bash scripts/start_eval_workers.sh --name shard_3 --redis-db 3 --work-dir /tmp/veritas_shard_3 --log-dir outputs/aes_test_run_002_parallel/logs/workers/shard_3

# 3. Execute shard evaluation concurrently
python3 scripts/test.py --config configs/test_run_002_shard_1.yaml --checkpoint models/checkpoints/run_002/final.pt --output-dir outputs/aes_test_run_002_parallel --run-id shard_1 --episodes all --device cpu --redis-db 1 --work-dir /tmp/veritas_shard_1 --evaluate-reset --prepare-cfg-calls --inline-functionality > outputs/aes_test_run_002_parallel/logs/test_shard_1.log 2>&1 &
python3 scripts/test.py --config configs/test_run_002_shard_2.yaml --checkpoint models/checkpoints/run_002/final.pt --output-dir outputs/aes_test_run_002_parallel --run-id shard_2 --episodes all --device cpu --redis-db 2 --work-dir /tmp/veritas_shard_2 --evaluate-reset --prepare-cfg-calls --inline-functionality > outputs/aes_test_run_002_parallel/logs/test_shard_2.log 2>&1 &
python3 scripts/test.py --config configs/test_run_002_shard_3.yaml --checkpoint models/checkpoints/run_002/final.pt --output-dir outputs/aes_test_run_002_parallel --run-id shard_3 --episodes all --device cpu --redis-db 3 --work-dir /tmp/veritas_shard_3 --evaluate-reset --prepare-cfg-calls --inline-functionality > outputs/aes_test_run_002_parallel/logs/test_shard_3.log 2>&1 &
wait

# 4. Merge shard outputs into a single report
python3 scripts/merge_eval_shards.py \
    --root outputs/aes_test_run_002_parallel \
    --manifest outputs/aes_test_run_002_parallel/shard_manifest.json \
    --shards shard_1 shard_2 shard_3

# 5. Stop worker pools
bash scripts/stop_eval_workers.sh --log-dir outputs/aes_test_run_002_parallel/logs/workers/shard_1
bash scripts/stop_eval_workers.sh --log-dir outputs/aes_test_run_002_parallel/logs/workers/shard_2
bash scripts/stop_eval_workers.sh --log-dir outputs/aes_test_run_002_parallel/logs/workers/shard_3
```

---

## Reproducing Paper Tables and Figures

The training and evaluation pipelines produce structured metrics, per-episode JSON execution traces, and generated adversarial binaries:

```text
outputs/aes_test_run_002/
|-- summary_metrics.json       # Consolidated evasion rates, valid execution rates, and step counts
|-- traces.json                # Complete trace collection across all evaluated samples
|-- traces/                    # Individual per-sample step-by-step diagnostic JSON files
`-- evasive_samples/           # Generated problem-space adversarial PE binaries
```

To reproduce the paper's experimental findings:

1. Execute evaluation on the holdout `Test/` dataset using `scripts/test.py` across the evaluated surrogate models.
2. Review `summary_metrics.json` for family-level evasion rates, valid execution rates, and average perturbation lengths.
3. Compare the generated adversarial binaries against baseline detection engines to verify black-box transferability.

---

## Main Results Summary

The following summary highlights key experimental findings reported in the paper:

- **Generation Coverage & Operational Validity**: Under the validity-constrained search policy, VERITAS produces finalized candidate outputs for 1,203 out of 1,794 holdout malware samples (67.06% Generation Coverage) with 100% joint operational validity ($V_{\mathrm{VER}} = 100\%$).
- **Cross-Target Attack Effectiveness**: Across nine held-out learning-based detectors, VERITAS achieves an unweighted mean Validity-Conditioned Attack Success Rate ($\mathrm{V\mbox{-}ASR}^{\mathrm{VER}}$) of **23.13%**, outperforming evaluated baselines (GAMMA 11.48%, OBFU-mal 10.38%, GAPGAN 1.12%, MAB-Malware 0.66%).
- **Target-Specific Attack Breadth**: Reaches **73.05%** $\mathrm{V\mbox{-}ASR}^{\mathrm{VER}}$ against sequence-oriented RTF-BERT (1,198 of 1,640 detected samples), **37.17%** on Multiclass M-Attn-Health, **34.47%** on Binary M-Attn-Health, **15.37%** on Multiclass MalConv, and **14.10%** on SeqConvAttn.
- **Operational Black-Box Transferability**: Variants optimized against a single local surrogate transfer effectively to unseen commercial operational targets without query access, achieving **23.16% Valid-ASR against ClamAV** and **14.89% against Microsoft Defender**, with a mean of **10.36%** across VirusTotal engine-detection-ratio (TER) thresholds.
- **Multi-Surface Transformation Synergy**: Multi-surface search coverage across headers, sections, code structure, CFG edges, and binary lifting is essential; restricting the action space to metadata-only or instruction-only subsets significantly degrades attack effectiveness across diverse detector representations.

---

## Backend System Architecture

The environment relies on an event-driven asynchronous pipeline with Redis as the central message broker.

### The 3-Stage Pipeline
**1. Extraction**: Four independent workers (`worker_api.py`, `worker_syscall.py`, `worker_angr.py`, `worker_static.py`) run in parallel, listening to Redis queues to extract features.
**2. Evaluation**: Triggered automatically when extraction completes.
  - **Detector Model**: Evaluates API chains, PE imports, and sections using the surrogate neural network.
  - **Functionality Evaluator**: A 3-layer waterfall filter checking execution viability (Integrity Gatekeeper → BinSim Semantic Prover → API Sequence Aligner).
**3. Reward Calculation**: Applies the parameterized mathematical objective function defined in `reward_calculator.py`.

### Sandbox Infrastructure
The system distributes sandbox analysis across 4 CAPEv2 KVM nodes using Atomic Round-Robin. Nodes are automatically cleaned via SSH between episodes to prevent contamination.

### Known Architectural Notes
- **angr Compatibility**: `worker_angr.py` spawns a subprocess pointing to the separate `func_venv` because `angr` is incompatible with Python 3.14. 

---

## Authors / Maintainers

*Redacted for anonymous review.*


---

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.
