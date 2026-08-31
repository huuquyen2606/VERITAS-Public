# DQEAF Reproduction (Paper-Faithful)

Paper: `Evading Anti-Malware Engines With Deep Reinforcement Learning` (IEEE Access, 2019)

This codebase reconstructs the training/testing workflow in the paper with the same RL logic:

- Environment observation: 513-d raw binary stream features
- Action space: `A = {ARBE, ARI, ARS, RS}`
- Reward and objective: equations (3), (4), (5), (8)
- Training loop: Algorithm 2
- Testing loop: Algorithm 3
- Prioritized replay sampling probability: `P(Xi) = pXi / sum_j pj`
- Double-network setup: `Q` and `Q_target`

## 1) Install

```bash
python -m pip install -r requirements.txt
```

## 2) One Config File

Main config file: `configure/config.ini`

Sections:

- `[Cuckoo]`: reserved fields (`ip`, `token`) for your sandbox workflow
- `[Dataset]`: dataset paths and caps
- `[Classifier]`: surrogate backend (`gbdt` or `lgbm_ember`) and model paths
- `[Train]`: paper default training hyperparameters
- `[Eval]`: evaluation settings
- `[Runtime]`: device
- `[Paths]`: output locations

## 3) Run (Short Commands)

Train (read all values from `configure/config.ini`):

```bash
python scripts/train.py
```

Evaluate:

```bash
python scripts/evaluate.py
```

## 4) Optional CLI Override

Any CLI arg overrides config file value, e.g.:

```bash
python scripts/train.py --config configure/config.ini --classifier-backend lgbm_ember --lgbm-model-path result.txt
```

```bash
python scripts/evaluate.py --config configure/config.ini --model-dir outputs/dqeaf_lgbm
```

## Notes

- The original paper used Chainer (Python 3.6 era). This reproduction ports only framework/runtime APIs to modern stack (`torch`, `gymnasium`) while preserving RL algorithm workflow.
- PE mutation actions are implemented via `lief` with the same four action categories from the paper.
- `scripts/evaluate.py` auto-detects classifier backend from `classifier_meta.json` in the model directory.
