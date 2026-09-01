# Framework Architecture

The `baseline_detectors` framework is built heavily around Object-Oriented principles to ensure absolute uniformity across all supported models. 

## The Core: `MalwareModelBase`

Located in `core.py`, the `MalwareModelBase` is an Abstract Base Class (ABC). Every model inside the `models/` directory **must** inherit from this class and implement the following abstract methods:

1. `train(self, train_paths, val_paths, label_mapping, experiment_name=None)`
2. `predict(self, input_path) -> dict`
3. `evaluate(self, test_path, output_dir=None) -> dict`
4. `load_weights(self, weights_path)`
5. `evade(self, technique_paths, label_mapping) -> dict`

Because every model is guaranteed to have these methods, `main.py` and `ensemble.py` can treat all models exactly the same, regardless of their underlying complexity (CNN vs Attention vs NLP-based).

## Shared Utilities (`utils.py`)

`utils.py` contains the shared infrastructure that powers the framework:

- **Smart Device Detection**: `get_optimal_device()` automatically falls back across CUDA -> XLA -> MPS -> CPU.
- **RAM-Safe Data Loading**: `ChunkRandomSampler` ensures that huge `.npz` datasets are loaded intelligently in chunks.
- **Experiment Management**: The `ExperimentManager` creates timestamped folders and saves config snapshots.
- **Live Logging**: The `Logger` ensures logs are simultaneously printed to the console and flushed to a file to survive crashes.
- **Standardized Visualizations**: Functions like `save_training_plot`, `generate_confusion_matrix`, and `plot_confidence_distribution` provide uniform analytics across all models.

## Adding a New Model

To add a new detector to the framework:

1. Create a new directory in `baseline_detectors/models/` (e.g., `my_new_model/`).
2. Implement your model architecture.
3. Create an `__init__.py` or main class file where your model class inherits from `MalwareModelBase` (imported from `baseline_detectors.core`).
4. Implement the 5 required abstract methods.
5. Register your new model in `baseline_detectors/__init__.py`:
   ```python
   from .models.my_new_model import MyNewModel

   MODEL_REGISTRY = {
       # ... existing models ...
       "my_new_model": MyNewModel
   }
   ```
6. Your model is now natively accessible via JSON configs in `main.py`!
