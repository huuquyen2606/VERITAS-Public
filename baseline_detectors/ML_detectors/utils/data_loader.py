import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,  # <--- ADD THIS
    recall_score,  # <--- ADD THIS
)


class NPZMalwareDataset(Dataset):
    def __init__(self, npz_filepath, label_mapping, feature_type="op_code"):
        """
        Loads the .npz dictionary pointers into memory without blowing up RAM.
        feature_type: choose between 'raw_byte', 'op_code', 'api_cuckoo', etc.
        """
        # Load pointer, allow_pickle=True is required for strings/object arrays
        self.data = np.load(npz_filepath, allow_pickle=True)
        self.labels = self.data["label"]
        self.names = self.data["name"]

        # Load the requested feature
        self.features = self.data[feature_type]
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 1. Trích xuất đặc trưng của MỘT mẫu duy nhất (Rất nhẹ cho RAM)
        raw_feature = self.features[idx]

        # --- FEATURE ENGINEERING GOES HERE ---
        # Example: Convert OpCodes to a dummy tensor (Replace with your actual NLP/CNN embedding logic)
        x_tensor = torch.zeros(100)  # Placeholder for the processed feature

        # 2. Xử lý Nhãn dựa trên label_mapping từ JSON
        string_label = self.labels[idx]
        y_label = self.label_mapping.get(string_label, 0)  # Default to 0 if unknown
        y_tensor = torch.tensor(y_label, dtype=torch.long)

        return x_tensor, y_tensor
