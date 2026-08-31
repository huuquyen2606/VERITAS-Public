#!/usr/bin/env python3
"""
Train a Gradient Boosted Decision Tree (GBDT) surrogate model for gym-malware.

Per the paper (Anderson et al., arXiv:1801.08917v2, Section 4):
  - Model: sklearn GradientBoostingClassifier
  - Features: PEFeatureExtractor (2350 dimensions)
  - Trained on malicious + benign PE samples
  - Threshold ~0.9 (~1% FPR at ~90% TPR)

Usage:
  python train_gbdt.py \
    --malware-dir Adv/Locker Adv/Mediyes Adv/Winwebsec Adv/Zbot Adv/Zeroaccess \
    --benign-dir Adv/Benign \
    --output-dir gym_malware/envs/utils/
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import traceback

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    roc_curve,
)

# Ensure gym_malware is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from gym_malware.envs.utils.pefeatures import PEFeatureExtractor


def iter_pe_files(dirs):
    """Yield absolute paths to PE files from one or more directories."""
    for d in dirs:
        d = os.path.abspath(d)
        if not os.path.isdir(d):
            print(f"[WARN] Not a directory, skipping: {d}")
            continue
        for fp in sorted(glob.glob(os.path.join(d, "*"))):
            if os.path.isfile(fp):
                yield fp


def extract_features_from_dirs(dirs, label, extractor, limit=0):
    """Extract feature vectors from all PE files in the given directories."""
    features = []
    labels = []
    errors = 0
    count = 0
    for filepath in iter_pe_files(dirs):
        if limit and count >= limit:
            break
        try:
            with open(filepath, "rb") as f:
                bytez = f.read()
            feats = extractor.extract(bytez)
            if len(feats) != extractor.dim:
                print(f"[WARN] Dim mismatch {len(feats)} != {extractor.dim}: {filepath}")
                errors += 1
                continue
            features.append(feats)
            labels.append(label)
            count += 1
            if count % 200 == 0:
                print(f"  ... extracted {count} samples (label={label})")
        except Exception:
            errors += 1
            if errors <= 5:
                traceback.print_exc()
            elif errors == 6:
                print("  ... suppressing further error tracebacks")
    print(f"  Extracted {count} samples (label={label}), {errors} errors")
    return features, labels


def export_tree_to_dict(tree):
    """Convert a single sklearn DecisionTreeRegressor to a serializable dict."""
    t = tree.tree_
    return {
        "children_left": t.children_left.tolist(),
        "children_right": t.children_right.tolist(),
        "feature": t.feature.tolist(),
        "threshold": t.threshold.tolist(),
        "value": [float(v[0][0]) for v in t.value],
    }


def export_gbdt_to_json(model, output_path):
    """Export a trained GradientBoostingClassifier to JSON for CustomGBDT."""
    # For binary classification with deviance loss, init is LogOddsEstimator
    # The init prediction is the log-odds of the positive class
    init_value = float(model.init_.class_prior_[1])
    # Convert to log-odds
    import math
    if init_value <= 0 or init_value >= 1:
        log_odds = 0.0
    else:
        log_odds = math.log(init_value / (1.0 - init_value))

    trees = []
    for i in range(model.n_estimators_):
        tree = model.estimators_[i, 0]  # binary classification: single tree per stage
        trees.append(export_tree_to_dict(tree))

    data = {
        "init_value": log_odds,
        "learning_rate": model.learning_rate,
        "trees": trees,
    }

    with open(output_path, "w") as f:
        json.dump(data, f)
    print(f"Exported JSON model to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train sklearn GradientBoostingClassifier for gym-malware (per paper)."
    )
    parser.add_argument(
        "--malware-dir",
        nargs="+",
        required=True,
        help="One or more directories containing malware PE files.",
    )
    parser.add_argument(
        "--benign-dir",
        nargs="+",
        required=True,
        help="One or more directories containing benign PE files.",
    )
    parser.add_argument(
        "--output-dir",
        default="gym_malware/envs/utils/",
        help="Directory to save model files (default: gym_malware/envs/utils/).",
    )
    parser.add_argument(
        "--malware-test-dir",
        nargs="*",
        default=[],
        help="Directories with malware PEs for testing.",
    )
    parser.add_argument(
        "--benign-test-dir",
        nargs="*",
        default=[],
        help="Directories with benign PEs for testing.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for testing (default: 0.2) if test dirs not provided.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of boosting stages (default: 100).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Max depth of individual trees (default: 3).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate / shrinkage (default: 0.1).",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="Min samples per leaf (default: 1).",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="Subsample ratio for stochastic gradient boosting (default: 1.0).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=123,
        help="Random seed (default: 123, same as gym-malware).",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=0,
        help="Max samples per class (0=no limit).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract features only, don't train.",
    )
    args = parser.parse_args()

    np.random.seed(args.random_seed)

    # Feature extraction
    extractor = PEFeatureExtractor()
    print(f"Feature extractor dimension: {extractor.dim}")
    assert extractor.dim == 2350, f"Expected 2350 dimensions, got {extractor.dim}"

    print("\n=== Extracting malware features ===")
    mal_feats, mal_labels = extract_features_from_dirs(
        args.malware_dir, label=1, extractor=extractor, limit=args.limit_per_class
    )

    print("\n=== Extracting benign features ===")
    ben_feats, ben_labels = extract_features_from_dirs(
        args.benign_dir, label=0, extractor=extractor, limit=args.limit_per_class
    )

    all_feats = np.array(mal_feats + ben_feats, dtype=np.float32)
    all_labels = np.array(mal_labels + ben_labels, dtype=np.int32)

    print(f"\nTotal samples: {len(all_labels)} (malware={sum(all_labels)}, benign={len(all_labels)-sum(all_labels)})")
    print(f"Feature shape: {all_feats.shape}")

    if args.dry_run:
        print("\n[DRY RUN] Feature extraction complete. Exiting without training.")
        return

    # Train/test split or explicit test dirs
    if args.malware_test_dir or args.benign_test_dir:
        print("\n=== Extracting explicit testing set ===")
        mal_test_feats, mal_test_labels = extract_features_from_dirs(
            args.malware_test_dir, label=1, extractor=extractor, limit=args.limit_per_class
        )
        ben_test_feats, ben_test_labels = extract_features_from_dirs(
            args.benign_test_dir, label=0, extractor=extractor, limit=args.limit_per_class
        )
        X_train = all_feats
        y_train = all_labels
        X_test = np.array(mal_test_feats + ben_test_feats, dtype=np.float32)
        y_test = np.array(mal_test_labels + ben_test_labels, dtype=np.int32)
    else:
        # Fallback to Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            all_feats, all_labels, test_size=args.test_size, random_state=args.random_seed, stratify=all_labels
        )
    print(f"Train: {len(y_train)} (mal={sum(y_train)}, ben={len(y_train)-sum(y_train)})")
    print(f"Test:  {len(y_test)} (mal={sum(y_test)}, ben={len(y_test)-sum(y_test)})")

    # Train
    print("\n=== Training GradientBoostingClassifier ===")
    print(f"  n_estimators={args.n_estimators}, max_depth={args.max_depth}, "
          f"learning_rate={args.learning_rate}, min_samples_leaf={args.min_samples_leaf}, "
          f"subsample={args.subsample}")

    model = GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_samples_leaf=args.min_samples_leaf,
        subsample=args.subsample,
        random_state=args.random_seed,
        verbose=1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    print("\n=== Evaluation ===")
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC: {auc:.4f}")

    # Find threshold at ~1% FPR (per paper)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    target_fpr = 0.01
    idx = np.searchsorted(fpr, target_fpr)
    if idx < len(thresholds):
        paper_threshold = thresholds[idx]
        paper_tpr = tpr[idx]
        paper_fpr = fpr[idx]
    else:
        paper_threshold = 0.9
        paper_tpr = tpr[-1]
        paper_fpr = fpr[-1]

    print(f"\nAt ~1% FPR:")
    print(f"  Threshold: {paper_threshold:.4f}")
    print(f"  FPR: {paper_fpr:.4f}")
    print(f"  TPR: {paper_tpr:.4f}")
    print(f"\nPaper uses threshold=0.9. Using 0.9 for classification report:")

    y_pred_09 = (y_prob >= 0.9).astype(int)
    print(classification_report(y_test, y_pred_09, target_names=["benign", "malware"]))

    # Save models
    os.makedirs(args.output_dir, exist_ok=True)

    # Save PKL
    import joblib
    pkl_path = os.path.join(args.output_dir, "gradient_boosting.pkl")
    joblib.dump(model, pkl_path)
    print(f"Saved PKL model to {pkl_path}")

    # Save JSON (for CustomGBDT)
    json_path = os.path.join(args.output_dir, "gradient_boosting.json")
    export_gbdt_to_json(model, json_path)

    # Save TXT (Evaluation Report)
    txt_path = os.path.join(args.output_dir, "gradient_boosting.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Gradient Boosting Evaluation Report ===\n")
        f.write(f"ROC-AUC: {auc:.4f}\n\n")
        f.write(f"At ~1% FPR:\n")
        f.write(f"  Threshold: {paper_threshold:.4f}\n")
        f.write(f"  FPR: {paper_fpr:.4f}\n")
        f.write(f"  TPR: {paper_tpr:.4f}\n\n")
        f.write("Classification Report (Threshold = 0.9):\n")
        f.write(classification_report(y_test, y_pred_09, target_names=["benign", "malware"]))
    print(f"Saved TXT report to {txt_path}")

    # Verify JSON matches PKL
    print("\n=== Verifying JSON/PKL equivalence ===")
    from gym_malware.envs.utils.gbdt_custom import CustomGBDT
    json_model = CustomGBDT(json_path)

    n_check = min(50, len(X_test))
    max_diff = 0.0
    for i in range(n_check):
        pkl_score = model.predict_proba(X_test[i:i+1])[0, 1]
        json_score = json_model.predict_proba(X_test[i:i+1])[0, 1]
        diff = abs(pkl_score - json_score)
        if diff > max_diff:
            max_diff = diff
    print(f"Max score difference across {n_check} samples: {max_diff:.10f}")
    if max_diff < 1e-6:
        print("✓ JSON and PKL models produce equivalent scores.")
    else:
        print(f"⚠ WARNING: Score differences detected (max={max_diff:.6f}). Review export logic.")

    print("\n=== Done ===")
    print(f"Output files:")
    print(f"  {pkl_path}")
    print(f"  {json_path}")
    print(f"  {txt_path}")
    print(f"\nTo use this model, ensure interface.py correctly loads from: {json_path}")


if __name__ == "__main__":
    main()
