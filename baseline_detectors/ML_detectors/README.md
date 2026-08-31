# Baseline Detectors Framework

Welcome to the **Baseline Detectors Framework**! This is a comprehensive, modular malware detection suite designed for researchers and security practitioners. It provides a robust architecture for training, evaluating, predicting, and testing evasion robustness against various malware detection models.

## Overview
This standalone package encapsulates all essential model architectures, a dynamic execution controller (`main.py`), and a FastAPI server (`server.py`). It is built to seamlessly transition between single-model evaluations and complex ensemble-based predictions.

### Key Features
- **Unified Interface:** All models implement the `MalwareModelBase` class, ensuring consistent `.train()`, `.predict()`, `.evaluate()`, and `.evade()` methods.
- **Ensemble Support:** Easily combine multiple models with custom weights to build powerful ensemble detectors.
- **Smart Resource Management:** Automatically detects available hardware (GPU/CUDA, TPU/XLA, MPS, or CPU) and allocates system resources efficiently to prevent OOM errors.
- **Evasion Testing:** Native support for evaluating model robustness against adversarial samples and evasion techniques.

## Supported Models
This framework includes several state-of-the-art architectures for static malware detection:

1. **MalConv (`malconv`)**: Standard CNN-based architecture for raw byte analysis.
2. **SeqConvAttn (`seqconvattn`)**: Sequence-based convolutional network with attention mechanisms.
3. **IMCFN (`imcfn`)**: Image-based malware classification network.
4. **Binary MalConv (`binary_malconv`)**: A variant of MalConv optimized for binary classification.
5. **MultiView CNN (`multiview_cnn`)**: Analyzes multiple representations of malware samples.
6. **MAttn Health (`m_attn_health` & `m_attn_health_multi`)**: Attention-based models with health monitoring features.
7. **RTF Cannie (`rtf_cannie`)**
8. **RTF Bert (`rtf_bert`)**

## Quickstart

### Prerequisites
Make sure you have the required dependencies installed (primarily `torch`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `psutil`, `fastapi`, and `uvicorn`). 

### Running the Controller
The primary entry point is `main.py`, which is driven entirely by JSON configuration files.

```bash
# Run a specific task using a config file
python -m baseline_detectors.main --config config.json
```

For more details on how to write these configurations and use the framework, please refer to the following documentation files:

- [HOW_TO_USE.md](./HOW_TO_USE.md): Instructions on running predictions, evaluations, and evasion testing.
- [TRAINING_AND_TUNING.md](./TRAINING_AND_TUNING.md): Guide on training new models and dataset formats.
- [ARCHITECTURE.md](./ARCHITECTURE.md): Deep dive into the framework's core design and how to add new models.
