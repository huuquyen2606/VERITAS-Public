from abc import ABC, abstractmethod
from .utils import get_optimal_device  # <-- Import our new smart detector


class MalwareModelBase(ABC):
    def __init__(self):
        # Every model will now automatically find the best hardware!
        self.device = get_optimal_device()

    """
    Abstract Base Class for all malware detection models in the framework.

    This class defines the standard interface that every model (e.g., MalConv,
    ResNet50, etc.) must implement. This ensures that the main CLI,
    the API Server, and the EnsembleManager can interact with any model uniformly,
    whether processing raw PE files or extracted .npz datasets.
    """

    @abstractmethod
    def train(self, train_paths, val_paths, label_mapping, experiment_name=None):
        """
        Train the model using its internal architecture and configuration.

        This method handles the entire training loop, including data loading,
        logging (to tensorboard/txt), checkpoint saving, and validation.
        It must be capable of routing data through the RAM-safe NPZ loader
        (from ./utils/) if an .npz file is provided, or a standard directory loader.

        Args:
            train_paths (list or str): Path(s) to the training dataset directories OR .npz files.
            val_paths (list or str): Path(s) to the validation dataset directories OR .npz files.
            label_mapping (dict): A dictionary mapping class names to integer labels.
                                  Example: {"Benign": 0, "Virus": 1}
            experiment_name (str, optional): A unique name for this training run.
                                             If None, a timestamped name is usually generated.

        Returns:
            None: The method should save artifacts to disk but return nothing.
        """
        pass

    @abstractmethod
    def predict(self, input_path) -> dict:
        """
        Perform inference on a single file, a folder of files, or an extracted .npz dataset.

        Args:
            input_path (str): Path to a single file (e.g., 'sample.exe'), a directory
                              containing files to scan, or a 'dataset.npz' file.

        Returns:
            dict: A dictionary containing the prediction scores and the final assigned label.
                  For .npz files, the key should be formatted as "filepath::hash".
                  Example:
                  {
                      "C:/malware/sample.exe": {  # Or "dataset.npz::abc123hash"
                          "Benign": 0.05,
                          "Virus": 0.95,
                          "final_label": "Virus"
                      }
                  }
        """
        pass

    @abstractmethod
    def evaluate(self, test_path, output_dir=None) -> dict:
        """
        Evaluate the model's performance on a labeled test dataset.

        This method calculates metrics (Accuracy, F1, Precision, Recall),
        generates a confusion matrix, and optionally plots confidence distributions.

        Args:
            test_path (str): Path to the directory or .npz file containing labeled test data.
                             (If a directory, structure should match training data).
            output_dir (str, optional): Directory where evaluation artifacts (plots,
                                        reports) should be saved. Defaults to internal config if None.

        Returns:
            dict: A dictionary containing the calculated metrics.
                  Example: {"accuracy": 0.95, "f1_score": 0.94, ...}
        """
        pass

    @abstractmethod
    def load_weights(self, weights_path):
        """
        Load pre-trained weights into the model architecture.

        This is required before calling `predict`, `evaluate`, or `evade`.

        Args:
            weights_path (str): Path to the saved model state dictionary (.pth file).
                                Should handle both full paths and paths relative
                                to the model's 'weights' directory.

        Raises:
            FileNotFoundError: If the weights file cannot be located.
        """
        pass

    @abstractmethod
    def evade(self, technique_paths, label_mapping) -> dict:
        """
        Test the model's robustness against specific adversarial evasion techniques.

        This method iterates through folders or .npz files of adversarial samples
        (organized by technique) and calculates how many successfully bypass the
        detector (i.e., are misclassified).

        Args:
            technique_paths (list): A list of paths to directories or .npz files containing
                                    adversarial samples. Each path typically represents one
                                    evasion technique (e.g., '.../adv_samples/UPX_packed').
            label_mapping (dict): The class mapping used during training. Crucial for
                                  determining if a prediction is "correct" (blocked)
                                  or "incorrect" (evaded).

        Returns:
            dict: A report dictionary summarizing the evasion success rate for each technique.
                  Structure:
                  {
                      "TechniqueName": {
                          "passed": int,   # Number of samples that successfully evaded
                          "total": int,    # Total number of samples tested
                          "rate": float    # Evasion rate (0.0 to 1.0)
                      },
                      ...
                  }
        """
        pass
