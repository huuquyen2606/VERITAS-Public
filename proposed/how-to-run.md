# VERITAS - Installation and Execution Guide

All commands should be executed from the root of the repository:
```bash
cd VERITAS  # Navigate to your cloned repository root
```

---

## 1. Environment Setup

### 1.1 Python Environment (Main Agent)
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 1.2 Backend Functionality Environment (`func_venv`)
The backend functionality workers (angr, triton) require a dedicated Python <= 3.12 environment:
```bash
python3 -m venv env/adv_RL_env/func_venv
env/adv_RL_env/func_venv/bin/python -m pip install --upgrade pip setuptools wheel
env/adv_RL_env/func_venv/bin/python -m pip install -r env/adv_RL_env/requirements_functionality.txt
```

### 1.3 Build C++ Smith-Waterman Extension
Install the extension for BOTH environments:
```bash
pip install ./env/adv_RL_env/examples/comparing_apis
env/adv_RL_env/func_venv/bin/python -m pip install ./env/adv_RL_env/examples/comparing_apis
```

### 1.4 Sanity Check
```bash
python3 -c "import torch, lief, angr, z3, triton, redis, fast_sw; print('All core dependencies imported successfully.')"
env/adv_RL_env/func_venv/bin/python -c "import angr, triton, z3, fast_sw; print('Backend dependencies imported successfully.')"
```

---

## 2. Offline Precomputation (Phase 0)

### 2.1 Benign Content Bank (Required)
Extracts benign code, data, imports, and stubs used by mutation actions.
```bash
python3 scripts/prepare_benign_content.py \
    --benign_dir data/dataset/Adv_agent/Benign \
    --output_dir data/benign_content
```
Verify output exists:
```bash
ls -lh data/benign_content/benign_bank.pkl
```

### 2.2 In-Place Code Randomization (Recommended for Action 15)
```bash
# Training Set
for family in Locker Mediyes Winwebsec Zbot Zeroaccess; do
    python3 scripts/precompute_code_randomize.py \
        --input-dir data/dataset/Adv_agent/$family \
        --output-dir data/precomputed/train/code-randomize \
        --seed 1337 --max-rewrites 200 --timeout-sec 180
done

# Test Set
for family in Locker Mediyes Winwebsec Zbot Zeroaccess; do
    python3 scripts/precompute_code_randomize.py \
        --input-dir data/dataset/Test/$family \
        --output-dir data/precomputed/test/code-randomize \
        --seed 1337 --max-rewrites 200 --timeout-sec 180
done
```

### 2.3 Code Translation via OLLVM (Recommended for Action 3)
```bash
RETDEC=utils/RetDec/bin/retdec-decompiler
OLLVM=utils/obfuscator_llvm_14/build/bin

# Training Set
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

# Test Set
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

## 3. Training the Model (Phase 1)

### 3.1 Configure CAPE Sandbox Nodes
Before running, you must provide your own CAPEv2 Sandbox node IPs, SSH credentials, and API Tokens for the evaluation workers.
Edit `assets/cape_nodes.json` and replace the placeholder `<YOUR_CAPE_NODE_X_IP>` and `<YOUR_SSH_PASSWORD>` with your actual setup.
Also, ensure you replace `<YOUR_CAPE_API_TOKEN>` in `env/adv_RL_env/malware_rl_system/workers/worker_api.py` and `worker_syscall.py`.

### 3.2 Start Microservices Backend
```bash
# Start Redis
redis-server --daemonize yes
redis-cli ping   # Should return PONG

# Start extraction workers (API, Syscall, Static, Angr)
cd env/adv_RL_env && bash start_workers.sh && cd ../..

# Start Functionality Evaluator listener daemon
bash scripts/start_functionality_evaluator.sh
```

### 3.3 Smoke Run (1 Episode Test)
```bash
python3 scripts/train.py \
    --config configs/train.yaml \
    --limit 1 \
    --evaluate-reset \
    --fail-fast
```

### 3.4 Full Training
```bash
python3 scripts/train.py \
    --config configs/train.yaml \
    --device cpu \
    --evaluate-reset \
    --prepare-cfg-calls \
    --inline-functionality \
    2>&1 | tee logs/train_full.log
```

Monitor training history in realtime:
```bash
tail -f logs/runs/run_002/train_history.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    print(f'ep={d[\"episode\"]} reward={d[\"total_reward\"]:.1f} evaded={d[\"evaded\"]} ER={d[\"evasion_rate\"]:.3f}')
"
```

---

## 4. Single-Process Evaluation (Phase 2)

Evaluate a trained checkpoint on the holdout test set:
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

View summary metrics:
```bash
python3 -m json.tool outputs/aes_test_run_002/summary_metrics.json
```

---

## 5. Parallel Multi-Shard Evaluation (Phase 3)

For fast evaluation across multi-core servers, partition the test dataset into isolated shards.

### 5.1 Verify Checkpoint
```bash
python3 -c '
from pathlib import Path
from scripts.rl_entrypoint_utils import load_yaml_config, choose_device, make_agent_kwargs, load_checkpoint
from agent.veritas_agent import VERITASAgent
cfg = load_yaml_config("configs/test.yaml")
device = choose_device("cpu")
agent = VERITASAgent(**make_agent_kwargs(cfg, device))
ckpt = load_checkpoint(Path("models/checkpoints/run_002/final.pt"), agent, device, load_optimizer=False)
print({"episode": ckpt.get("episode"), "global_step": ckpt.get("global_step"), "num_actions": agent.num_actions, "num_atoms": agent.num_atoms})
'
```

### 5.2 Create Shards
```bash
python3 scripts/create_eval_shards.py \
    --input-root data/dataset/Test \
    --output-root data/dataset/Test_shards/run_002 \
    --families Locker Mediyes Winwebsec Zbot Zeroaccess \
    --shards 3 \
    --mode symlink \
    --manifest outputs/aes_test_run_002_parallel/shard_manifest.json
```

### 5.3 Flush Redis DBs & Start Worker Shards
```bash
redis-cli -n 1 FLUSHDB
redis-cli -n 2 FLUSHDB
redis-cli -n 3 FLUSHDB

# Shard 1
bash scripts/start_eval_workers.sh \
    --name shard_1 --redis-db 1 \
    --cape-config assets/cape_nodes_eval_shard_1.json \
    --work-dir /tmp/veritas_eval_shard_1 \
    --log-dir outputs/aes_test_run_002_parallel/logs/workers/shard_1

# Shard 2
bash scripts/start_eval_workers.sh \
    --name shard_2 --redis-db 2 \
    --cape-config assets/cape_nodes_eval_shard_2.json \
    --work-dir /tmp/veritas_eval_shard_2 \
    --log-dir outputs/aes_test_run_002_parallel/logs/workers/shard_2

# Shard 3
bash scripts/start_eval_workers.sh \
    --name shard_3 --redis-db 3 \
    --cape-config assets/cape_nodes_eval_shard_3.json \
    --work-dir /tmp/veritas_eval_shard_3 \
    --log-dir outputs/aes_test_run_002_parallel/logs/workers/shard_3
```

### 5.4 Run Parallel Shards
Run each shard in separate tmux windows or background subshells:

```bash
# Shard 1
python3 scripts/test.py \
    --config configs/test_run_002_shard_1.yaml \
    --checkpoint models/checkpoints/run_002/final.pt \
    --output-dir outputs/aes_test_run_002_parallel \
    --run-id shard_1 --episodes all --device cpu --redis-db 1 \
    --work-dir /tmp/veritas_eval_shard_1 \
    --evaluate-reset --prepare-cfg-calls --inline-functionality \
    2>&1 | tee outputs/aes_test_run_002_parallel/logs/test_shard_1.log &

# Shard 2
python3 scripts/test.py \
    --config configs/test_run_002_shard_2.yaml \
    --checkpoint models/checkpoints/run_002/final.pt \
    --output-dir outputs/aes_test_run_002_parallel \
    --run-id shard_2 --episodes all --device cpu --redis-db 2 \
    --work-dir /tmp/veritas_eval_shard_2 \
    --evaluate-reset --prepare-cfg-calls --inline-functionality \
    2>&1 | tee outputs/aes_test_run_002_parallel/logs/test_shard_2.log &

# Shard 3
python3 scripts/test.py \
    --config configs/test_run_002_shard_3.yaml \
    --checkpoint models/checkpoints/run_002/final.pt \
    --output-dir outputs/aes_test_run_002_parallel \
    --run-id shard_3 --episodes all --device cpu --redis-db 3 \
    --work-dir /tmp/veritas_eval_shard_3 \
    --evaluate-reset --prepare-cfg-calls --inline-functionality \
    2>&1 | tee outputs/aes_test_run_002_parallel/logs/test_shard_3.log &

wait
```

### 5.5 Resume a Stopped Shard (If Interrupted)
To resume a specific shard without re-evaluating completed samples:
```bash
python3 scripts/test.py \
    --config configs/test_run_002_shard_3.yaml \
    --checkpoint models/checkpoints/run_002/final.pt \
    --output-dir outputs/aes_test_run_002_parallel \
    --run-id shard_3 --episodes all --device cpu --redis-db 3 \
    --work-dir /tmp/veritas_eval_shard_3 \
    --evaluate-reset --prepare-cfg-calls --inline-functionality \
    --resume-existing \
    2>&1 | tee -a outputs/aes_test_run_002_parallel/logs/test_shard_3.log
```

### 5.6 Merge Shard Results
```bash
python3 scripts/merge_eval_shards.py \
    --root outputs/aes_test_run_002_parallel \
    --manifest outputs/aes_test_run_002_parallel/shard_manifest.json \
    --shards shard_1 shard_2 shard_3
```

### 5.7 Stop Worker Shards
```bash
bash scripts/stop_eval_workers.sh --log-dir outputs/aes_test_run_002_parallel/logs/workers/shard_1
bash scripts/stop_eval_workers.sh --log-dir outputs/aes_test_run_002_parallel/logs/workers/shard_2
bash scripts/stop_eval_workers.sh --log-dir outputs/aes_test_run_002_parallel/logs/workers/shard_3
```

---

## 6. Pre-Run Checklist & Troubleshooting

### Checklist
1. Virtual environment is active (`source venv/bin/activate`).
2. Redis is reachable (`redis-cli ping` returns `PONG`).
3. C++ extension `fast_sw` is installed in current python environment.
4. Benign bank artifact `data/benign_content/benign_bank.pkl` exists.
5. Model weights exist at `env/adv_RL_env/malware_rl_system/weights/m_attn_health_multi.pth`.
6. Sandbox server/nodes are accessible.

### Troubleshooting
- **Redis connection error**: Run `redis-server --daemonize yes`.
- **Barrier timeout error**: Verify extraction workers are running with `ps aux | grep worker_`.
- **Evaluation hang**: Ensure Functionality Evaluator daemon is running or supply `--inline-functionality` to run in-process.
- **CUDA OOM**: Switch to CPU inference by appending `--device cpu`.
