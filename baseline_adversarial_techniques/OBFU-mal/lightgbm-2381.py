import os
import glob
import sys
import numpy as np
import lightgbm as lgb
import matplotlib
# Set backend to Agg for headless server environments (prevents GUI errors)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# ==========================================
# PATH CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = SCRIPT_DIR
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "LIGHTGBMs")
MODEL_FILENAME = "result.txt"
MODEL_PATH = os.path.join(OUTPUT_DIR, MODEL_FILENAME)

try:
    import ember
except ImportError:
    sys.path.append(SCRIPT_DIR)
    import ember

import lief
if hasattr(lief, 'logging'):
    lief.logging.set_level(lief.logging.LEVEL.CRITICAL)

# ==========================================
# PART 1: FEATURE MAPPING (NAME GENERATION)
# ==========================================
def get_ember_feature_names():
    """Generates a list of names for the 2381 features of EMBER v2"""
    names = []
    names += [f"ByteHist_{i}" for i in range(256)]           # 0-255
    names += [f"ByteEntropy_{i}" for i in range(256)]        # 256-511
    names += [f"StringInfo_{i}" for i in range(104)]         # 512-615
    names += [f"GenInfo_{i}" for i in range(10)]             # 616-625
    names += [f"Header_{i}" for i in range(62)]              # 626-687
    names += [f"Section_{i}" for i in range(255)]            # 688-942
    names += [f"Import_{i}" for i in range(1280)]            # 943-2222
    names += [f"Export_{i}" for i in range(128)]             # 2223-2350
    
    # Padding if missing (to ensure exactly 2381)
    if len(names) < 2381:
        names += [f"Extra_{i}" for i in range(2381 - len(names))]
    return names[:2381]

# ==========================================
# PART 2: FEATURE EXTRACTION (MULTI-THREADED)
# ==========================================
def worker_extract(args):
    path, label = args
    try:
        with open(path, "rb") as f:
            file_data = f.read()
        if not file_data: return None

        extractor = ember.PEFeatureExtractor(2, print_feature_warning=False)
        feature_vector = extractor.feature_vector(file_data)
        return (np.array(feature_vector, dtype=np.float32), label)
    except:
        return None

def load_dataset(benign_folder, malware_parent_folder, desc="Loading Data"):
    """
    Load and process Train or Test sets
    Args:
        benign_folder: Path to folder containing benign samples
        malware_parent_folder: Parent folder containing multiple malware family subfolders
                               (e.g., Dataset/Adv containing Locker/, Mediyes/, etc.)
    """
    tasks = []
    
    # Load Benign files (label=0)
    if os.path.exists(benign_folder):
        benign_files = glob.glob(os.path.join(benign_folder, "*"))
        tasks.extend([(f, 0) for f in benign_files if os.path.isfile(f)])
        print(f"[+] Found {len([f for f in benign_files if os.path.isfile(f)])} benign samples")
    
    # Load Malware files (label=1) from all subdirectories except Benign
    if os.path.exists(malware_parent_folder):
        malware_count = 0
        for subfolder in os.listdir(malware_parent_folder):
            subfolder_path = os.path.join(malware_parent_folder, subfolder)
            # Skip the Benign folder to avoid duplicates
            if subfolder.lower() == "benign" or not os.path.isdir(subfolder_path):
                continue
            
            malware_files = [f for f in glob.glob(os.path.join(subfolder_path, "**", "*"), recursive=True) 
                           if os.path.isfile(f)]
            tasks.extend([(f, 1) for f in malware_files])
            print(f"[+] Found {len(malware_files)} samples in {subfolder}")
            malware_count += len(malware_files)
        print(f"[+] Total malware samples: {malware_count}")

    if not tasks: 
        print("[!] No samples found!")
        return None, None

    print(f"[*] Total samples to process: {len(tasks)}")
    num_cores = cpu_count()
    X_data, y_data = [], []
    with Pool(processes=num_cores) as pool:
        for res in tqdm(pool.imap(worker_extract, tasks), total=len(tasks), desc=desc):
            if res:
                X_data.append(res[0])
                y_data.append(res[1])
    
    print(f"[*] Successfully extracted {len(X_data)} feature vectors")
    return np.array(X_data), np.array(y_data)

# ==========================================
# PART 3: MAIN PIPELINE (TRAIN + EVAL + PLOT)
# ==========================================
def run_all():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    # ----------------------------------------
    # STEP 1: TRAINING
    # ----------------------------------------
    print("\n" + "="*40 + "\n STEP 1: TRAINING \n" + "="*40)
    X_train_full, y_train_full = load_dataset(
        os.path.join(DATASET_DIR, "Adv_detector", "Benign"),
        os.path.join(DATASET_DIR, "Adv_detector"),
        "Extracting Train Set"
    )
    if X_train_full is None:
        print("[!] No Training data found.")
        return

    # ---- Validation split 80/20 (stratified để giữ tỷ lệ benign/malware) ----
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )
    print(f"[*] Train: {X_train.shape[0]} samples | Val: {X_val.shape[0]} samples")

    feature_names = get_ember_feature_names()
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data   = lgb.Dataset(X_val,   label=y_val,   feature_name=feature_names, reference=train_data)

    params = {
        "boosting": "gbdt",
        "objective": "binary",
        "metric": "auc",

        # 1. Complexity control — chống overfit
        "num_leaves": 2048,           # restored standard EMBER 
        "max_depth": 15,              # restored standard EMBER
        "min_data_in_leaf": 50,       # restored standard EMBER

        # 2. Randomization — đa dạng hóa cây
        "feature_fraction": 0.5,   # 50%
        "bagging_fraction": 0.8,   # 80%
        "bagging_freq": 1,         # REQUIRED: bagging_fraction chỉ hoạt động khi freq > 0

        "learning_rate": 0.05,
        "verbose": -1,
        "num_threads": -1,
    }

    num_boost_round     = 500
    early_stopping_rounds = 50

    print(f"[*] Training LightGBM on {X_train.shape[0]} samples (val={X_val.shape[0]})...")
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )
    print(f"[*] Best iteration: {gbm.best_iteration}")
    gbm.save_model(MODEL_PATH)
    print(f"[SUCCESS] Model saved at: {MODEL_PATH}")

    # ----------------------------------------
    # STEP 2: EVALUATION (TEST SET)
    # ----------------------------------------
    
    print("\n" + "="*40 + "\n STEP 2: EVALUATION (TEST SET) \n" + "="*40)
    X_test, y_test = load_dataset(
        os.path.join(DATASET_DIR, "Test", "Benign"),
        os.path.join(DATASET_DIR, "Test"),
        "Extracting Test Set"
    )

    if X_test is not None and len(X_test) > 0:
        print(f"[*] Predicting on {X_test.shape[0]} test samples...")
        y_prob = gbm.predict(X_test)
        y_pred = [1 if p >= 0.5 else 0 for p in y_prob] # 0.5 Threshold

        # Metrics calculation
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print("\n" + "-"*35)
        print(" CLASSIFICATION RESULTS REPORT")
        print("-"*35)
        print(f" Accuracy : {acc*100:.2f}%")
        print(f" F1-Score : {f1:.4f}")
        print("\n" + classification_report(y_test, y_pred, target_names=["Benign", "Malware"]))

        # Plot Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Benign", "Malware"], yticklabels=["Benign", "Malware"])
        plt.title('Confusion Matrix (Test Set)')
        plt.ylabel('Actual (True Label)')
        plt.xlabel('Predicted (Predicted Label)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
        plt.close()
        print(f"-> Saved Confusion Matrix: {os.path.join(OUTPUT_DIR, 'confusion_matrix.png')}")
    else:
        print("[!] No Test data found, skipping evaluation.")

    # ----------------------------------------
    # STEP 3: VISUALIZATION
    # ----------------------------------------
    
    print("\n" + "="*40 + "\n STEP 3: VISUALIZATION \n" + "="*40)
    
    # 3.1 Feature Importance (with mapped names)
    plt.figure(figsize=(12, 6))
    lgb.plot_importance(gbm, max_num_features=20, importance_type='gain', height=0.5, title='Top 20 Features (Mapped Names)')
    plt.tight_layout()
    img_path1 = os.path.join(OUTPUT_DIR, 'feature_importance.png')
    plt.savefig(img_path1)
    plt.close()
    print(f"-> Saved Feature Importance: {img_path1}")

    # 3.2 Tree Visualization
    try:
        if os.system("dot -V") == 0:
            plt.figure(figsize=(24, 24))
            lgb.plot_tree(gbm, tree_index=0, figsize=(24, 24), show_info=['split_gain'])
            img_path2 = os.path.join(OUTPUT_DIR, 'tree_visualization.png')
            plt.savefig(img_path2, dpi=200)
            plt.close()
            print(f"-> Saved Tree Plot: {img_path2}")
    except:
        pass

if __name__ == "__main__":
    run_all()
    print("\n" + "="*40 + "\n PIPELINE COMPLETED \n" + "="*40)