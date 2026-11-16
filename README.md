# SeedLab: Machine Learning Reproducibility & Seed Variability Dashboard

**"An Experimental Study of Random Seed Variability and Its Impact on the Reproducibility of Machine Learning Model Performance"**

A high-contrast **Dark Mode** visual analytics platform designed to visualize and compare random seed variability across multiple machine learning algorithm families.

---

## 🎯 1. Presentation Architecture

- **Experiment Execution (Google Colab):** Dataset generation/loading, stratified 80:20 partitioning, and training across seeds (`0, 1, 2, 3, 4`) are executed and demonstrated live in Google Colab using [`colab_experiment.py`](file:///c:/Users/yashb/OneDrive/Desktop/ML%20mini%20project/colab_experiment.py).
- **Visualization Platform (Streamlit Dashboard):** The dashboard ingests the exported `seedlab_results.csv` and presents interactive comparison matrices, deviation tables, sensitivity rankings, and trajectory charts.

---

## 🧭 2. Dashboard Features (Single-Page Visual Analytics)

1. **Top Data Bar:**
   - Drag-and-drop CSV uploader for live Colab results.
   - 1-click **"⚡ Load Pre-run Colab Results"** button for instant demonstration.
2. **Algorithm View Selector:**
   - **`📊 All Models (Comparison Table & Matrix)`** (unified view).
   - Drill-down views: **Random Forest**, **Neural Network (MLP)**, and **Logistic Regression**.
3. **Core Analysis Components in Unified View:**
   - **🔍 Unified Seed-by-Seed Comparison Table:** Direct side-by-side comparison of test accuracy for every seed with green (best) and yellow (lowest) highlights.
   - **📈 Seed Fluctuation Table (Deviation from Model Mean):** Shows exact $+ / -$ percentage deviations from the model's overall mean.
   - **🏆 Sensitivity & Spread Ranking Table:** Mean accuracy, standard deviation ($ddof=1$), and total seed spread ($\Delta$) side-by-side.
   - **Interactive Visualizations:** Comparative accuracy trajectory line chart + sensitivity ranking bar chart.
   - **Key Comparative Insight:** Clear summary for the viva presentation.
4. **Individual Model Drill-Down:**
   - 5 dedicated metric cards: Mean Accuracy, Std Dev, Lowest Run, Highest Run, Seed Gap.
   - Model-specific accuracy and latency charts.
   - Detailed run table.
5. **CSV Export:** **📥 Download Full Results CSV** button.

---

## 💻 3. Google Colab Script (`colab_experiment.py`)

Run [`colab_experiment.py`](file:///c:/Users/yashb/OneDrive/Desktop/ML%20mini%20project/colab_experiment.py) in [Google Colab](https://colab.research.google.com):
1. Creates fixed stratified 80:20 split.
2. Trains **Random Forest**, **Neural Network (MLP)**, and **Logistic Regression** across seeds `0, 1, 2, 3, 4`.
3. Exports `seedlab_results.csv` and triggers automatic browser download.
4. Upload `seedlab_results.csv` directly into the SeedLab dashboard!

---

## 🚀 4. How to Run Dashboard Locally

```bash
# 1. Activate the Python virtual environment
.\venv\Scripts\Activate.ps1

# 2. Launch the Streamlit dashboard
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🎤 5. Two-Minute Presentation Flow (Colab + Dashboard)

1. **In Google Colab (1 minute):**
   > *"We executed our experimental training pipeline in Google Colab. We locked an 80:20 stratified split and trained three distinct algorithm families (Random Forest, Neural Network, and Logistic Regression) five times, changing only the random seed from 0 to 4. We then exported `seedlab_results.csv`."*

2. **In the Dashboard (1 minute):**
   > *(Show [http://localhost:8501](http://localhost:8501))*
   > *"We import the results directly into our SeedLab dashboard. In this unified comparison matrix, we can see:
   > - The **Neural Network** has the highest seed sensitivity ($\pm 0.166\%$, $0.390\%$ spread) due to random initial synaptic weights on non-convex loss surfaces.
   > - **Random Forest** has moderate sensitivity because ensemble averaging dampens individual tree volatility.
   > - **Logistic Regression** has virtually zero variance ($0.010\%$) because its convex loss surface has only one global minimum.
   > This visually proves that seed sensitivity is mathematically dictated by the model's loss surface."*
