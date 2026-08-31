from secml.array import CArray
from secml_malware.attack.blackbox.c_wrapper_phi import CWrapperPhi

class MyCustomModelWrapper(CWrapperPhi):
    def __init__(self, custom_model):
        """
        Args:
            custom_model: Your actual detection model adapter (RemoteModelAdapter).
        """
        self.classifier = custom_model

        # STATE INJECTION: Stores the label of the current malware being processed.
        self.current_original_label = None

    def extract_features(self, x: CArray):
        # We operate on raw bytes, so no feature extraction is needed here.
        return x

    def predict(self, x: CArray, return_decision_function: bool = True):
        """
        Custom predict function that bridges CArray -> RemoteModelAdapter
        """
        # --- CRITICAL FIX: Added the underscore to _forward(x) ---
        scores = self.classifier._forward(x)

        # 2. Extract probabilities
        # scores is a 2D array: [[prob_benign, prob_malware]]
        prob_malware = scores[0, 1].item()

        # 3. Determine the discrete label (1 for Malware, 0 for Benign)
        label = 1 if prob_malware >= 0.5 else 0
        labels = CArray([label])

        # 4. Return in the format expected by secml
        if return_decision_function:
            return labels, scores
        return labels