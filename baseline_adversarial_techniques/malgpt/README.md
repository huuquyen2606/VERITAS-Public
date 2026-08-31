# MalGPT Evasion Technique

**MalGPT** is an adversarial technique leveraging language models to generate and inject stealthy adversarial code sequences into PE files to evade detection.

This directory contains the standalone module for running the MalGPT technique.

## Installation

Ensure you have installed the specific dependencies for this technique:

```bash
pip install -r ../requirements.txt
```

## Usage

The runner script is `run_malgpt.py`. It requires a JSON configuration file specifying the command (`train` or `generate`).

```bash
python run_malgpt.py --config config.json
```

### Training MalGPT

If you want to train MalGPT to learn representations of benign code:

```json
{
    "command": "train",
    "models": ["malgpt"],
    "name": "MalGPT_Training_Run",
    "dataset": "/path/to/benign.npz",
    "save_path": "./weights/"
}
```

### Generating Adversarial Samples

To inject adversarial blocks into a target dataset:

```json
{
    "command": "generate",
    "models": ["malgpt"],
    "name": "MalGPT_Generation_Run",
    "dataset": "/path/to/malware.npz",
    "load_path": "/path/to/trained_malgpt.pth",
    "output_dir": "./Functional_Adv_Samples"
}
```
