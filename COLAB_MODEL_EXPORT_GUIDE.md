# SeedLab AI: Google Colab Model Export Guide

This guide explains how to export trained machine learning estimators, probability calibrations (`CalibratedClassifierCV`), and authentic performance metrics from Google Colab into **SeedLab AI**.

---

## 1. Scientific Standards & Zero Fabrication Policy

SeedLab AI strictly prohibits fabricated, assumed, or placeholder numbers. Every reported metric, probability, and matrix must originate from authentic model execution:

1. **Prediction Probabilities:** Extracted directly from `.predict_proba()` without artificial inflation or capping.
2. **Probability Calibration:** Uncalibrated raw probabilities can be overconfident or poorly calibrated. We train probability calibrators (`CalibratedClassifierCV`) using cross-validation on the training set to prevent test data leakage.
3. **Brier Score Validation:** Calibration quality is quantitatively validated using the multi-class Brier score:
   $$\text{Brier Score} = \frac{1}{N} \sum_{i=1}^N \sum_{k=0}^{K-1} (P_{ik} - Y_{ik})^2$$
   A lower Brier score confirms improved probability reliability.
4. **Held-Out Test Set Separation:** An unseen test set (2,500 samples) is strictly reserved to compute final test accuracy, macro precision, macro recall, macro F1-score, and the confusion matrix.

---

## 2. Artifacts Produced for SeedLab AI

| File | Purpose | Application Location |
| :--- | :--- | :--- |
| `seedlab_results.csv` | Performance metrics across random seeds (Acc, Prec, Rec, F1, Latency) | Uploaded to **Experiment Dashboard** |
| `seedlab_model_artifacts.zip` | Serialized calibrated estimators and complete test metadata | Loaded on **Model Prediction** |

### Inside `seedlab_model_artifacts.zip`:
- `random_forest.joblib` (Calibrated Random Forest estimator)
- `mlp.joblib` (Calibrated MLP Neural Network estimator)
- `logistic_regression.joblib` (Calibrated Logistic Regression estimator)
- `metadata.json` (Held-out test metrics, Brier scores, confusion matrices, and preprocessing parameters)

---

## 3. Running the Complete Script in Google Colab

Simply copy and run [`colab_experiment.py`](file:///c:/Users/yashb/OneDrive/Desktop/ML%20mini%20project/colab_experiment.py) in a Google Colab notebook cell.

It automatically:
1. Loads the official 70,000-sample MNIST benchmark dataset from OpenML/Kaggle.
2. Creates an 80:20 stratified split (10,000 training, 2,500 held-out test).
3. Trains Random Forest, MLP, and Logistic Regression across 5 random seeds (`[0, 1, 2, 3, 4]`).
4. Selects the representative estimator closest to each model family's mean F1-score.
5. Calibrates probability estimates using `CalibratedClassifierCV(method='sigmoid', cv=3)` on the training partition.
6. Evaluates calibrated models on the held-out test partition (computing authentic accuracy, precision, recall, F1, Brier scores, and confusion matrices).
7. Packages `seedlab_model_artifacts.zip` and automatically triggers browser downloads.

---

## 4. Clear Conceptual Distinctions for Viva Presentation

When defending this work in a college viva, clearly distinguish these three concepts:

1. **Seed Variability (Experiment Dashboard):**
   - Measures how changing random initialization seeds (`0, 1, 2, 3, 4`) causes performance fluctuation on the exact same data split.
   - Shows standard deviation ($ddof=1$) and min-max spread ($\Delta$).

2. **Held-Out Generalization (Model Prediction - Test Evaluation Section):**
   - Evaluates the final fitted and calibrated model on 2,500 unseen test samples.
   - Reports authentic Accuracy, Macro F1, Precision, Recall, Brier Score, and Confusion Matrix.

3. **Individual Drawing Probability (Model Prediction - Interactive Canvas):**
   - The calibrated likelihood assigned to a single handwritten stroke.
   - *Scientific Note:* A high probability indicates the model strongly favors that class given the feature vector; it does not guarantee ground-truth correctness.
