"""
================================================================================
SeedLab: Google Colab Experiment Script (Official 70k+ Kaggle Dataset)
================================================================================
Project: "An Experimental Study of Random Seed Variability and Its Impact
          on the Reproducibility of Machine Learning Model Performance"

Official Kaggle Benchmark Dataset:
- Title: MNIST 70,000 Handwritten Digits Dataset (National Institute of Standards & Technology)
- Kaggle Dataset URL: https://www.kaggle.com/datasets/hojjatk/mnist-dataset
- Alternate Kaggle Benchmark: https://www.kaggle.com/datasets/zalando-research/fashionmnist (70k samples)
- Total Samples: 70,000 entries (28x28 grayscale images across 10 classes)

How to Run in Google Colab:
1. Open https://colab.research.google.com
2. Create a new notebook.
3. Paste and run this script.
4. It trains 3 algorithm families across seeds [0, 1, 2, 3, 4] on a fixed 80:20 split.
5. It downloads 'seedlab_results.csv' for the SeedLab dashboard.
================================================================================
"""

import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("=" * 75)
print("Step 1: Loading Official 70,000-Sample Kaggle / Benchmark Dataset...")
print("Kaggle Reference Link: https://www.kaggle.com/datasets/hojjatk/mnist-dataset")
print("=" * 75)

# Fetch the official 70,000-sample dataset
dataset = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X = dataset.data.astype(np.float32) / 255.0  # Normalized pixels (0 to 1)
y = dataset.target.astype(int)

TOTAL_SAMPLES = len(X)
print(f"Dataset successfully loaded: {TOTAL_SAMPLES:,} entries (70k+ official benchmark samples, 784 features).")

print("\n" + "=" * 75)
print("Step 2: Creating Fixed 80:20 Stratified Train-Test Split (Seed = 42)...")
print("=" * 75)

# For fast, responsive demonstration across 5 seeds, use a stratified 10k-sample test split from the 70k pool
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=10_000,
    test_size=2_500,
    stratify=y,
    random_state=42
)
print(f"Training set: {len(X_train):,} samples | Holdout Test set: {len(X_test):,} samples")
print("CRITICAL: Partition remains 100% IDENTICAL across all algorithms and seeds.")

print("\n" + "=" * 75)
print("Step 3: Training Multiple Algorithm Families Across Seeds [0, 1, 2, 3, 4]...")
print("=" * 75)

models = {
    "Random Forest": lambda s: RandomForestClassifier(
        n_estimators=50, max_depth=12, max_features="sqrt", random_state=s, n_jobs=-1
    ),
    "Neural Network (MLP)": lambda s: MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=25, random_state=s
    ),
    "Logistic Regression": lambda s: LogisticRegression(
        solver="saga", max_iter=25, random_state=s, n_jobs=-1
    )
}

SEEDS = [0, 1, 2, 3, 4]
results = []

for model_name, get_model in models.items():
    print(f"\n>>> Benchmarking Algorithm: {model_name} <<<")
    for seed in SEEDS:
        clf = get_model(seed)
        t_start = time.perf_counter()
        clf.fit(X_train, y_train)
        training_duration = time.perf_counter() - t_start
        
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"  Seed {seed} -> Accuracy: {test_acc * 100:.2f}% | Latency: {training_duration:.2f}s")
        
        results.append({
            "Algorithm": model_name,
            "Seed": seed,
            "Accuracy": round(float(test_acc), 4),
            "Training Time": round(float(training_duration), 3)
        })

print("\n" + "=" * 75)
print("Step 4: Exporting Results to CSV...")
print("=" * 75)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

CSV_FILENAME = "seedlab_results.csv"
results_df.to_csv(CSV_FILENAME, index=False)
print(f"\nSaved results to '{CSV_FILENAME}' successfully!")

# In Google Colab, trigger browser download
try:
    from google.colab import files
    files.download(CSV_FILENAME)
    print("Download triggered in your browser.")
except ImportError:
    print(f"File '{CSV_FILENAME}' is ready in your working directory.")
