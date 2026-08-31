# How to Use the Baseline Detectors

The `baseline_detectors` framework is controlled via a central CLI script: `main.py`. Instead of passing numerous command-line arguments, you define your execution parameters in a simple JSON configuration file.

## General Command Structure

```bash
python -m baseline_detectors.main --config path/to/config.json [--name optional_experiment_name]
```

## Modes of Operation

The framework operates in two distinct modes depending on your configuration:
1. **Single Model Mode:** Specify a single `"model"` string.
2. **Ensemble Mode:** Specify a list of `"models"`.

---

## 1. Prediction (`predict`)

Run inference on a single file, a directory of PE files, or an extracted `.npz` dataset.

### Single Model Prediction Config
```json
{
    "command": "predict",
    "model": "malconv",
    "weights": "experiments/2026-08-22_Run/best_model.pth",
    "input": "path/to/suspicious_file.exe"
}
```

### Ensemble Prediction Config
```json
{
    "command": "predict",
    "models": ["malconv", "seqconvattn"],
    "weights": {
        "malconv": "path/to/malconv_weights.pth",
        "seqconvattn": "path/to/seqconvattn_weights.pth"
    },
    "input": "path/to/dataset.npz"
}
```

---

## 2. Evaluation (`evaluate`)

Evaluate a model's performance on a labeled test dataset. Generates metrics, confusion matrices, and confidence distribution plots.

### Evaluation Config
```json
{
    "command": "evaluate",
    "model": "seqconvattn",
    "weights": "path/to/weights.pth",
    "test_dir": "path/to/test_dataset.npz",
    "output_dir": "results/eval_output"
}
```

---

## 3. Evasion Testing (`evade`)

Test the model's robustness against specific adversarial evasion techniques. The framework will calculate how many adversarial samples successfully bypassed the detector.

### Evasion Config
```json
{
    "command": "evade",
    "model": "malconv",
    "weights": "path/to/weights.pth",
    "map": "path/to/label_map.json",
    "adv_dirs": [
        "path/to/evasion/technique_1.npz",
        "path/to/evasion/technique_2.npz"
    ]
}
```

*Note: The `--map` (label mapping) is required so the framework knows which classes are considered "Benign" vs "Malware" to correctly determine if an evasion was successful.*

---

## 4. Server Mode (API)

The framework includes a FastAPI server (`server.py`) that exposes the models as an HTTP API. This is particularly useful for external tools (like adversarial generation scripts such as GAMMA) that need to query the model continuously.

### Starting the Server

```bash
python -m baseline_detectors.server --host 0.0.0.0 --port 8000
```

### Server Endpoints

1. **Check Status**
   ```bash
   curl http://127.0.0.1:8000/status
   ```
   *Returns the current state of the server (idle, or which models are loaded).*

2. **Load a Model (or Ensemble)**
   Before predicting, you must load a model into the server's memory.
   ```bash
   curl -X POST http://127.0.0.1:8000/load -H "Content-Type: application/json" -d '{
       "model": "malconv",
       "single_weights": "path/to/malconv_weights.pth"
   }'
   ```
   *For an ensemble, use the `"models"` and `"weights"` keys similar to the config files.*

3. **Predict**
   ```bash
   curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
       "input": "path/to/suspicious_file.exe"
   }'
   ```
   *Returns the prediction results and confidence scores.*

4. **Evade**
   ```bash
   curl -X POST http://127.0.0.1:8000/evade -H "Content-Type: application/json" -d '{
       "adv_dirs": ["path/to/evasion_dir"],
       "map": {"benign": 0, "malware": 1}
   }'
   ```

---

## 5. Detailed Logging (Per-File Results)

The framework automatically saves detailed, per-file analysis logs for every dataset you process. This allows you to inspect the exact prediction, confidence scores, and status for individual samples.

### AI Detectors (`main.py`)
- **Predictions (`predict`):** Saves a detailed JSON file named `per_file_prediction_results.json` in your current directory, mapping each scanned file to its prediction breakdown.
- **Evasion Tests (`evade`):** Saves a detailed JSON file named `per_file_evasion_results.json` in your current directory, mapping each sample to its true label and evasion status for each model.

### Heuristic Detectors
The heuristic detectors save their detailed per-file results locally inside their respective script directories:
- **Windows Defender:** `baseline_detectors/heuristic_detectors/windows_defender/defender_results.json`
- **ClamAV:** `baseline_detectors/heuristic_detectors/clamav/clamav_results.json`
