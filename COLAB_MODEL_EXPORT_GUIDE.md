# SeedLab AI: Google Colab Model Export Guide

This guide explains how to export your trained machine learning estimators and experiment metrics from Google Colab to power the **SeedLab AI** application.

---

## 1. Overview of Artifacts Needed by SeedLab AI

The SeedLab application performs **inference-only** (zero in-app training). It requires two files exported from Google Colab:

| File | Purpose | Application Location |
| :--- | :--- | :--- |
| `seedlab_results.csv` | Performance metrics across random seeds (Acc, Prec, Rec, F1, Time) | Uploaded to **Experiment Dashboard** |
| `seedlab_model_artifacts.zip` | Serialized scikit-learn estimators and preprocessing metadata | Loaded on **Model Prediction** |

### Inside `seedlab_model_artifacts.zip`:
- `random_forest.joblib` (Trained Random Forest estimator)
- `mlp.joblib` (Trained MLP Neural Network estimator)
- `logistic_regression.joblib` (Trained Logistic Regression estimator)
- `metadata.json` (Specification of input shapes, class labels, and normalization)

---

## 2. Option A: Using the All-in-One Training & Export Script

If you want to run the full benchmark experiment from scratch, simply copy and paste the contents of [`colab_experiment.py`](file:///c:/Users/yashb/OneDrive/Desktop/ML mini project/colab_experiment.py) into a Google Colab notebook cell.

It automatically:
1. Loads the official 70,000-sample MNIST benchmark dataset from OpenML/Kaggle.
2. Creates an 80:20 stratified split.
3. Trains Random Forest, MLP, and Logistic Regression across 5 random seeds (`[0, 1, 2, 3, 4]`).
4. Selects the representative estimator closest to each model family's mean F1-score.
5. Generates `metadata.json`.
6. Packages and triggers browser downloads of both `seedlab_results.csv` and `seedlab_model_artifacts.zip`.

---

## 3. Option B: Adding Artifact Export to an Existing Colab Notebook

If you **already ran** your training loop in Colab, add the following export cell at the end of your notebook.

### Colab Export Cell Code

```python
# ==============================================================================
# SEEDLAB AI: ARTIFACT EXPORT STEP
# Paste this cell into your existing Google Colab Notebook
# ==============================================================================

import os
import json
import zipfile
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
from google.colab import files

# Create models directory
os.makedirs("models", exist_ok=True)

# ------------------------------------------------------------------------------
# STEP 1: Map your trained estimators
# If you retained fitted models per seed, choose the representative run closest to the mean F1:
# Alternatively, if you have fitted single estimators, assign them directly:
# ------------------------------------------------------------------------------

# Example 1: If you have fitted estimator variables:
# rf_model = your_fitted_random_forest
# mlp_model = your_fitted_mlp
# lr_model = your_fitted_logistic_regression

# Save each estimator using joblib
joblib.dump(rf_model, "models/random_forest.joblib", compress=3)
joblib.dump(mlp_model, "models/mlp.joblib", compress=3)
joblib.dump(lr_model, "models/logistic_regression.joblib", compress=3)
print("Saved estimator .joblib files.")

# ------------------------------------------------------------------------------
# STEP 2: Create metadata.json
# Defines input shapes and preprocessing rules for SeedLab's inference engine
# ------------------------------------------------------------------------------
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
    "model_files": {
        "Random Forest": "random_forest.joblib",
        "MLP Neural Network": "mlp.joblib",
        "Logistic Regression": "logistic_regression.joblib"
    },
    "export_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
}

with open("models/metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print("Saved models/metadata.json")

# ------------------------------------------------------------------------------
# STEP 3: Package into seedlab_model_artifacts.zip
# ------------------------------------------------------------------------------
ZIP_FILENAME = "seedlab_model_artifacts.zip"
with zipfile.ZipFile(ZIP_FILENAME, "w", zipfile.ZIP_DEFLATED) as z:
    z.write("models/random_forest.joblib", arcname="random_forest.joblib")
    z.write("models/mlp.joblib", arcname="mlp.joblib")
    z.write("models/logistic_regression.joblib", arcname="logistic_regression.joblib")
    z.write("models/metadata.json", arcname="metadata.json")

print(f"Packaged {ZIP_FILENAME} successfully.")

# ------------------------------------------------------------------------------
# STEP 4: Trigger Browser Downloads
# ------------------------------------------------------------------------------
# Download the experiment CSV (if your results dataframe is named results_df):
if 'results_df' in locals():
    results_df.to_csv("seedlab_results.csv", index=False)
    files.download("seedlab_results.csv")
    print("Downloading seedlab_results.csv...")

files.download(ZIP_FILENAME)
print(f"Downloading {ZIP_FILENAME}...")
```

---

## 4. How SeedLab Selects the Representative Model

When training across multiple seeds (e.g. seeds 0, 1, 2, 3, 4):
1. **Selection Rule:** For each model family, compute the mean Macro F1-score across all seeds:
   $$\bar{F}_1 = \frac{1}{N} \sum_{i=1}^N F_{1, \text{seed}_i}$$
2. Select the specific estimator whose validation F1-score has the minimum absolute deviation from $\bar{F}_1$:
   $$\text{Selected Seed} = \arg\min_i \left| F_{1, \text{seed}_i} - \bar{F}_1 \right|$$
3. This guarantees that the deployed model exhibits **typical, representative performance** rather than extreme "seed luck" or an uncharacteristic outlier.

---

## 5. Preprocessing Alignment

SeedLab's live drawing canvas performs the exact transformation required by the trained estimators:
- **Input:** 280×280 RGB canvas drawing.
- **Bounding Box Crop:** Removes peripheral blank space around the drawn digit.
- **Aspect-Ratio Preserving Resize:** Resizes the digit to fit within a 20×20 bounding box.
- **Center of Mass Shift:** Computes center of mass via `scipy.ndimage.center_of_mass` and translates the digit so its center aligns with $(13.5, 13.5)$ on a 28×28 grid.
- **Scaling:** Values scaled to $[0.0, 1.0]$ matching standard MNIST format.
- **Flattening:** Reshaped to `(1, 784)` float tensor.
