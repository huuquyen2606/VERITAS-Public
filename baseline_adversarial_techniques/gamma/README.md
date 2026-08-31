# GAMMA Evasion Technique

The **GAMMA** (Genetic Adversarial Malware Modification Algorithm) technique optimizes adversarial modifications in the structure of malware files to bypass static detectors. 

This directory is standalone and can be executed without the parent framework.

## Installation

Ensure you have installed the specific dependencies for this technique:

```bash
pip install -r ../requirements.txt
```

*(Note: `secml-malware` is required for this technique.)*

### End-to-End Execution Flow (Using MalConv Surrogate)

GAMMA relies on querying a surrogate model (typically MalConv) over a local network API to iteratively improve its adversarial samples.

**Step 1: Start the Surrogate Model Server**
Open a terminal and start the `baseline_detectors` server. This exposes the MalConv model API that GAMMA will communicate with.

```bash
# From the project root
python -m baseline_detectors.server --host 0.0.0.0 --port 8000
```

**Step 2: Configure GAMMA**
Create a `config.json` for GAMMA. Note how `server_url` points to the server you just started, and `model_config` tells the server to load `binary_malconv`.

```json
{
    "dataset_dir": "/path/to/malware/directory",
    "benign_dir": "/path/to/benign/directory",
    "output_dir": "/path/to/output_adversarial/directory",
    "server_url": "http://127.0.0.1:8000",
    "model_config": {
        "model": "binary_malconv",
        "single_weights": "/absolute/path/to/malconv_weights.pth"
    }
}
```

*Note: The script will automatically instruct the server to load the model and weights defined in `model_config` upon connecting.*

**Step 3: Run the Attack**
In a new terminal, execute the evasion script. GAMMA will extract sections from the `benign_dir`, inject them into malware samples in `dataset_dir`, and repeatedly query the server to check if the modification successfully evades MalConv.

```bash
python start_evasion.py --config config.json
```
