# Training and Tuning Guide

Training new models in the `baseline_detectors` framework is fully automated, complete with live logging, experiment folder generation, and metric visualization.

## Training a Model

To train a model, use the `train` command in your JSON configuration.

### Training Config Example

```json
{
    "command": "train",
    "model": "malconv",
    "dataset": ["path/to/Train_Part1.npz", "path/to/Train_Part2.npz"],
    "val_dataset": "path/to/Validation.npz",
    "map": "path/to/label_map.json",
    "name": "MalConv_Finetune"
}
```

### Configuration Details:
- **`dataset`**: A path or list of paths to your training data (directories or `.npz` files).
- **`val_dataset`**: (Optional) Path(s) to validation data. If omitted, it will use the training dataset for validation.
- **`map`**: Required for training. A JSON file mapping string labels to integer classes (e.g., `{"Benign": 0, "Malware": 1}`).
- **`name`**: (Optional) A custom name for your experiment folder. This can also be overridden via the CLI `--name` flag.

### What Happens During Training?
1. An experiment directory is created in the model's specific folder (e.g., `baseline_detectors/models/malconv/experiments/YYYY-MM-DD_HH-MM_MalConv_Finetune/`).
2. A snapshot of your config is saved.
3. Live logging begins (`experiment_log.txt`).
4. Upon completion, a training plot (`training_plot.png`) is generated showing Loss and Accuracy curves over epochs.

## Dataset Formatting (.npz)

The framework is highly optimized for RAM-safe `.npz` datasets. When using `.npz` files, the framework's `ChunkRandomSampler` loads data in chunks, preventing disk thrashing and out-of-memory crashes.

A valid `.npz` dataset should contain the following arrays:
- `data` or `X`: The feature data (e.g., raw bytes).
- `label` or `y`: The ground truth labels.
- `name`: (Optional) The original filenames or hashes, used for granular reporting.

## Tuning and Resource Limits

If you encounter Memory Errors (OOM), you can tweak the global resource limits inside `main.py`.

At the top of `main.py`, you will find `RESOURCE_CONFIG`:
```python
RESOURCE_CONFIG = {
    "max_cpu_ram_gb": total_ram_gb,
    "max_cpu_cores": total_cores,
    "force_cpu_mode": False,
    "gpu_vram_limit": 1.0,
}
```
- Lower `gpu_vram_limit` (e.g., `0.5`) to restrict PyTorch to 50% of VRAM.
- Set `"force_cpu_mode": True` to completely bypass the GPU if necessary.
