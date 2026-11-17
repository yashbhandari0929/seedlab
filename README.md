# SeedLab AI: Machine Learning Reproducibility & Live Digit Prediction

**"An Experimental Study of Random Seed Variability and Its Impact on the Reproducibility of Machine Learning Model Performance"**

A modern dark-mode machine learning platform combining empirical seed reproducibility analytics with **automatic best model selection** and **live handwritten digit recognition**.

---

## 🧭 Navigation Architecture (Vertical Left Sidebar)

The application provides exactly two distinct navigation views from the vertical left sidebar:

1. **🔬 Experiment Dashboard:**
   - Multi-seed experiment CSV upload and reactive status badge.
   - Unified seed-by-seed accuracy matrix across Random Forest, MLP Neural Network, and Logistic Regression.
   - Comprehensive performance matrix (Mean Accuracy, Best/Worst Run, Spread $\Delta$, Std Dev $ddof=1$, Average Latency).
   - Seed fluctuation table (deviation from model mean).
   - Comparative accuracy trajectories and seed sensitivity ranking charts.
   - Viva-ready plain English insights and conclusions.
   - Full CSV results download.

2. **✍️ Model Prediction:**
   - **Automatic Dynamic Model Selection:** Ranks models by mean Macro F1-Score (fallback to accuracy if missing), F1 stability (std dev), accuracy, and latency.
   - **Model Selection Summary Card:** Displays winning model, primary metric, accuracy, F1 score, stability, latency, and detailed rationale.
   - **Compact 3-Model Comparison:** Explains why the selected model won over other contenders.
   - **Interactive Drawing Canvas (280×280 px):** Smooth freedraw canvas with dedicated *Predict Digit* and *Clear Canvas* buttons.
   - **MNIST Preprocessing Pipeline Preview:** Inspects the aspect-ratio preserved, center-of-mass aligned 28×28 tensor fed to the model.
   - **Live Prediction Results:** Prominent predicted digit badge, top confidence percentage, and full 10-class (0–9) Plotly probability bar chart.
   - **Collapsible 3-Model Cross-Comparison:** Compares inferences from Random Forest, MLP, and Logistic Regression on the drawn digit.
   - **Prediction History Table:** Stores and displays the last 5 user predictions with timestamps and a clear history button.

---

## ⚙️ Model Artifacts & Architecture

The application strictly separates **training** from **inference**:
- **Zero In-App Training:** All models are trained in Google Colab or offline scripts.
- **Model Directory (`models/`):**
  - `random_forest.joblib` (Random Forest Classifier)
  - `mlp.joblib` (Multi-Layer Perceptron Neural Network)
  - `logistic_regression.joblib` (Logistic Regression Classifier)
  - `metadata.json` (Specification of input shapes, class labels, and normalization)
  - `seedlab_model_artifacts.zip` (Portable artifact bundle)
- **Artifact Upload Fallback:** If local files are not detected, users can upload `seedlab_model_artifacts.zip` directly on the Model Prediction page.

---

## 🔬 Google Colab Export Workflow

To train and export models using Google Colab:
1. Open [`colab_experiment.py`](file:///c:/Users/yashb/OneDrive/Desktop/ML%20mini%20project/colab_experiment.py) and run it in Google Colab.
2. It trains all three algorithm families across seeds `0, 1, 2, 3, 4` on the official Kaggle MNIST dataset.
3. It selects the representative estimator closest to each family's mean F1 score.
4. It triggers automatic browser downloads of `seedlab_results.csv` and `seedlab_model_artifacts.zip`.
5. For complete integration details, consult [`COLAB_MODEL_EXPORT_GUIDE.md`](file:///c:/Users/yashb/OneDrive/Desktop/ML%20mini%20project/COLAB_MODEL_EXPORT_GUIDE.md).

---

## 🚀 Running SeedLab AI Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Streamlit
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.
