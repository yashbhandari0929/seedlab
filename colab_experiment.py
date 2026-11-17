"""
================================================================================
SeedLab AI: Google Colab Multi-Metric Experiment & Model Artifact Export Script
================================================================================
Project: "An Experimental Study of Random Seed Variability and Its Impact
          on the Reproducibility of Machine Learning Model Performance"

Official Kaggle Benchmark Dataset:
- Title: MNIST 70,000 Handwritten Digits Dataset
- Kaggle Dataset URL: https://www.kaggle.com/datasets/hojjatk/mnist-dataset
- Total Samples: 70,000 entries (28x28 grayscale images across 10 classes)

Outputs Generated for SeedLab AI:
1. 'seedlab_results.csv' (Accuracy, Precision, Recall, F1 Score, Training Time across 5 seeds)
2. 'seedlab_model_artifacts.zip' containing:
   - random_forest.joblib
   - mlp.joblib
   - logistic_regression.joblib
   - metadata.json

How to Run in Google Colab:
1. Open https://colab.research.google.com
2. Create a new notebook.
3. Paste and run this script.
4. When finished, both 'seedlab_results.csv' and 'seedlab_model_artifacts.zip'
   will be automatically downloaded to your machine.
================================================================================
"""

import os
import time
import json
import zipfile
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("=" * 80)
print("Step 1: Loading Official 70,000-Sample Kaggle / Benchmark Dataset (MNIST 784)...")
print("Kaggle Reference Link: https://www.kaggle.com/datasets/hojjatk/mnist-dataset")
print("=" * 80)

dataset = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X = dataset.data.astype(np.float32) / 255.0  # Normalized pixels in [0.0, 1.0]
y = dataset.target.astype(int)

print(f"Total dataset samples loaded: {len(X):,} entries (784 features, 10 classes: 0-9).")

print("\n" + "=" * 80)
print("Step 2: Creating Fixed Stratified Train-Test Split (Seed = 42)...")
print("=" * 80)

# Stratified split: 10,000 training, 2,500 holdout test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=10_000,
    test_size=2_500,
    stratify=y,
    random_state=42
)
print(f"Training partition: {len(X_train):,} samples | Holdout Test partition: {len(X_test):,} samples")

print("\n" + "=" * 80)
print("Step 3: Benchmarking Algorithm Families Across Seeds [0, 1, 2, 3, 4]...")
print("=" * 80)

models_config = {
    "Random Forest": lambda s: RandomForestClassifier(
        n_estimators=60, max_depth=14, max_features="sqrt", random_state=s, n_jobs=-1
    ),
    "MLP Neural Network": lambda s: MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=30, random_state=s
    ),
    "Logistic Regression": lambda s: LogisticRegression(
        solver="saga", max_iter=25, random_state=s, n_jobs=-1
    )
}

SEEDS = [0, 1, 2, 3, 4]
results = []
fitted_models = {name: [] for name in models_config}

for model_name, get_model in models_config.items():
    print(f"\n>>> Running Multi-Seed Evaluation for: {model_name} <<<")
    for seed in SEEDS:
        clf = get_model(seed)
        t_start = time.perf_counter()
        clf.fit(X_train, y_train)
        latency = time.perf_counter() - t_start

        y_pred = clf.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

        print(f"  Seed {seed} | Acc: {acc*100:.2f}% | F1: {f1*100:.2f}% | Latency: {latency:.2f}s")

        results.append({
            "Algorithm": model_name,
            "Seed": seed,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
            "Training Time": round(latency, 3)
        })

        fitted_models[model_name].append({
            "seed": seed,
            "estimator": clf,
            "f1": f1,
            "acc": acc
        })

print("\n" + "=" * 80)
print("Step 4: Exporting seedlab_results.csv...")
print("=" * 80)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

CSV_FILENAME = "seedlab_results.csv"
results_df.to_csv(CSV_FILENAME, index=False)
print(f"\nSuccessfully generated '{CSV_FILENAME}'")

print("\n" + "=" * 80)
print("Step 5: Selecting Representative Estimators & Exporting Artifacts...")
print("=" * 80)

os.makedirs("models", exist_ok=True)

artifact_map = {
    "Random Forest": "random_forest.joblib",
    "MLP Neural Network": "mlp.joblib",
    "Logistic Regression": "logistic_regression.joblib"
}

# Rule: Pick representative estimator closest to each family's mean F1-score
for model_name, file_name in artifact_map.items():
    runs = fitted_models[model_name]
    mean_f1 = np.mean([r["f1"] for r in runs])
    best_run = min(runs, key=lambda r: abs(r["f1"] - mean_f1))
    
    out_path = os.path.join("models", file_name)
    joblib.dump(best_run["estimator"], out_path, compress=3)
    print(f"Saved {model_name} (Representative Seed {best_run['seed']}, F1={best_run['f1']:.4f}) -> {out_path}")

# Write metadata.json
metadata = {
    "project": "SeedLab AI",
    "dataset": "MNIST Handwritten Digits",
    "dataset_source": "https://www.kaggle.com/datasets/hojjatk/mnist-dataset",
    "num_classes": 10,
    "class_labels": list(range(10)),
    "input_shape": [1, 784],
    "image_dimensions": [28, 28],
    "pixel_scaling": "0_to_1",
    "preprocessing": {
        "flatten": True,
        "center_of_mass": True,
        "normalization": "divide_by_255",
        "invert_colors": False
    },
    "model_files": artifact_map,
    "export_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
}

with open(os.path.join("models", "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print("Saved models/metadata.json")

# Create seedlab_model_artifacts.zip
ZIP_FILENAME = "seedlab_model_artifacts.zip"
with zipfile.ZipFile(ZIP_FILENAME, "w", zipfile.ZIP_DEFLATED) as z:
    for fname in list(artifact_map.values()) + ["metadata.json"]:
        fpath = os.path.join("models", fname)
        z.write(fpath, arcname=fname)
print(f"Packaged '{ZIP_FILENAME}' successfully!")

print("\n" + "=" * 80)
print("Step 6: Triggering Automatic Downloads...")
print("=" * 80)

try:
    from google.colab import files
    files.download(CSV_FILENAME)
    time.sleep(1)
    files.download(ZIP_FILENAME)
    print("Colab browser downloads triggered for CSV and model artifacts ZIP!")
except ImportError:
    print(f"Local environment detected: '{CSV_FILENAME}' and '{ZIP_FILENAME}' are ready in your directory.")
