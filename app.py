"""
SeedLab: Interactive Results & Analysis Dashboard (Dark Mode).

Project: "An Experimental Study of Random Seed Variability and Its Impact
          on the Reproducibility of Machine Learning Model Performance"

Visualization & Analysis Platform:
- Reads results CSV exported from Google Colab.
- Unified Seed-by-Seed Cross-Algorithm Comparison Matrix & Deviation Analysis.
- Performance Matrix & Simple Insights & Conclusions.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

SAMPLE_CSV_PATH = APP_DIR / "data" / "seedlab_results.csv"

# ---------------------------------------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SeedLab | ML Reproducibility Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load Custom Dark CSS
css_path = APP_DIR / "assets" / "style.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "uploaded_df" not in st.session_state:
    if SAMPLE_CSV_PATH.exists():
        try:
            st.session_state.uploaded_df = pd.read_csv(SAMPLE_CSV_PATH)
            st.session_state.data_source_label = "Multi-Algorithm Benchmark (Pre-run Colab Data)"
        except Exception:
            st.session_state.uploaded_df = None
            st.session_state.data_source_label = ""
    else:
        st.session_state.uploaded_df = None
        st.session_state.data_source_label = ""

if "data_source_label" not in st.session_state:
    st.session_state.data_source_label = ""


# ---------------------------------------------------------------------------
# Top Header & Brand Bar
# ---------------------------------------------------------------------------
col_nav_brand, col_nav_meta = st.columns([3.0, 1.2])

with col_nav_brand:
    st.markdown("""
    <div class="brand-title-group">
        <div class="brand-icon-box">SL</div>
        <div>
            <div class="brand-title-text">SeedLab</div>
            <div class="brand-subtitle-text">Machine Learning Reproducibility & Seed Variability Dashboard</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_nav_meta:
    has_data = st.session_state.uploaded_df is not None
    status_msg = "Results Loaded" if has_data else "Awaiting CSV"
    status_color = "#34D399" if has_data else "#94A3B8"
    st.markdown(
        f'<div style="text-align: right; padding-top: 6px; font-size: 13px; color: {status_color}; font-weight: 600;">'
        f'● {status_msg}'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown('<hr style="border: 0; border-top: 1px solid #334155; margin: 8px 0 20px 0;" />', unsafe_allow_html=True)


# ===========================================================================
# RESULTS & ANALYSIS (PRIMARY VIEW)
# ===========================================================================
st.markdown('<div class="page-header-title">Experiment Results & Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-header-desc">'
    'Compare model performance across random seeds, analyze statistical variation, '
    'and inspect the all-in-one seed comparison table across algorithms.'
    '</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# A. Upload / Demo Data Bar
# ---------------------------------------------------------------------------
st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
col_upload_left, col_upload_right = st.columns([2.5, 1.2])

with col_upload_left:
    uploaded_file = st.file_uploader(
        "Upload experiment CSV (from Google Colab)",
        type=["csv"],
        help="Expected headers: Seed, Accuracy, Training Time (and optional Algorithm)",
        label_visibility="collapsed"
    )

with col_upload_right:
    if st.button("⚡ Load Pre-run Colab Results", use_container_width=True):
        if SAMPLE_CSV_PATH.exists():
            try:
                sample_df = pd.read_csv(SAMPLE_CSV_PATH)
                st.session_state.uploaded_df = sample_df
                st.session_state.data_source_label = "Multi-Algorithm Benchmark (Pre-run Colab Data)"
                st.success("Loaded experimental data.")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading sample: {e}")

# Process Uploaded CSV
validation_error = None
if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        if raw_df.empty:
            validation_error = "The uploaded CSV file is empty."
        else:
            col_map = {}
            for col in raw_df.columns:
                c = str(col).strip().lower()
                if c in ["algorithm", "model", "classifier"]:
                    col_map[col] = "Algorithm"
                elif c in ["seed", "random_seed", "random seed"]:
                    col_map[col] = "Seed"
                elif c in ["accuracy", "acc", "test_accuracy"]:
                    col_map[col] = "Accuracy"
                elif c in ["training time", "training_time", "train_time", "time", "time (s)"]:
                    col_map[col] = "Training Time"
            raw_df = raw_df.rename(columns=col_map)
            
            required = ["Seed", "Accuracy", "Training Time"]
            missing = [c for c in required if c not in raw_df.columns]
            if missing:
                validation_error = f"Missing column(s): {', '.join(missing)}. Headers must include Seed, Accuracy, Training Time."
            else:
                raw_df["Seed"] = pd.to_numeric(raw_df["Seed"], errors="coerce")
                raw_df["Accuracy"] = pd.to_numeric(raw_df["Accuracy"], errors="coerce")
                raw_df["Training Time"] = pd.to_numeric(raw_df["Training Time"], errors="coerce")
                
                if raw_df.isnull().values.any():
                    validation_error = "Uploaded file contains non-numeric or missing values."
                else:
                    st.session_state.uploaded_df = raw_df
                    st.session_state.data_source_label = f"Uploaded: {uploaded_file.name}"
    except Exception as e:
        validation_error = f"Failed to parse CSV: {e}"

if validation_error:
    st.error(f"⚠️ {validation_error}")
    st.session_state.uploaded_df = None

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Display Content: Empty State OR Live Analysis
# ---------------------------------------------------------------------------
df = st.session_state.uploaded_df

if df is None or df.empty:
    st.markdown("""
    <div class="empty-upload-box">
        <div style="font-size: 36px; margin-bottom: 8px;">📂</div>
        <div class="empty-upload-title">No experiment results loaded yet</div>
        <div class="empty-upload-text">
            Click <b>'⚡ Load Pre-run Colab Results'</b> above for an instant demonstration, 
            or upload your own <code>seedlab_results.csv</code> file exported from Google Colab.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Normalize Accuracy to percentage
    analysis_df = df.copy()
    is_ratio = analysis_df["Accuracy"].max() <= 1.0
    multiplier = 100.0 if is_ratio else 1.0
    analysis_df["Accuracy_Pct"] = analysis_df["Accuracy"] * multiplier

    has_multi_algo = "Algorithm" in analysis_df.columns and analysis_df["Algorithm"].nunique() > 1

    selected_algo = None
    if has_multi_algo:
        algo_list = ["📊 All Models (Comparison Table & Matrix)"] + sorted(analysis_df["Algorithm"].unique().tolist())
        col_sel1, col_sel2 = st.columns([2, 3])
        with col_sel1:
            selected_algo = st.selectbox(
                "Select View:",
                options=algo_list,
                index=0,
                help="Choose the unified all-in-one comparison table or drill down into a single model."
            )

    # ---------------------------------------------------------------------------
    # Case A: All Models (Unified All-in-One Comparison Matrix & Charts)
    # ---------------------------------------------------------------------------
    if has_multi_algo and selected_algo == "📊 All Models (Comparison Table & Matrix)":
        st.caption(f"Comparing **{analysis_df['Algorithm'].nunique()} algorithms** side-by-side across all seeds — {st.session_state.data_source_label}")

        # =======================================================================
        # 1. PRIMARY ALL-IN-ONE SEED COMPARISON TABLE (PIVOT MATRIX)
        # =======================================================================
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔍 Unified Seed-by-Seed Accuracy Matrix</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size: 13px; color: #94A3B8; margin-bottom: 12px;">'
            'Compare test accuracy for each random seed side-by-side across all algorithms. '
            'Notice how each seed affects each model differently.'
            '</div>',
            unsafe_allow_html=True
        )

        # Build Pivot Accuracy Table
        pivot_acc = analysis_df.pivot(index="Seed", columns="Algorithm", values="Accuracy_Pct")
        pivot_acc.index = [f"Seed {int(s)}" for s in pivot_acc.index]
        pivot_acc.index.name = "Random Seed"

        def highlight_column_extrema(s):
            is_max = s == s.max()
            is_min = s == s.min()
            return [
                'background-color: rgba(52, 211, 153, 0.25); color: #34D399; font-weight: 700;' if v_max else
                'background-color: rgba(251, 191, 36, 0.25); color: #FBBF24; font-weight: 700;' if v_min else ''
                for v_max, v_min in zip(is_max, is_min)
            ]

        styled_pivot = pivot_acc.style.apply(highlight_column_extrema, axis=0).format("{:.3f}%")
        st.dataframe(styled_pivot, use_container_width=True)
        st.caption("🟢 Green = Highest accuracy seed for that model | 🟡 Yellow = Lowest accuracy seed for that model.")
        st.markdown('</div>', unsafe_allow_html=True)

        # =======================================================================
        # 2. COMPREHENSIVE PERFORMANCE MATRIX
        # =======================================================================
        summary_records = []
        for algo, group in analysis_df.groupby("Algorithm"):
            m_acc = group["Accuracy_Pct"].mean()
            s_acc = group["Accuracy_Pct"].std(ddof=1) if len(group) > 1 else 0.0
            min_a = group["Accuracy_Pct"].min()
            max_a = group["Accuracy_Pct"].max()
            rng_a = max_a - min_a
            min_t = group["Training Time"].min()
            max_t = group["Training Time"].max()
            avg_t = group["Training Time"].mean()
            
            # Sensitivity label
            if rng_a < 0.05:
                sens = "🟢 Zero / Negligible"
            elif rng_a < 0.30:
                sens = "🟡 Low / Moderate"
            else:
                sens = "🔴 High Sensitivity"

            summary_records.append({
                "Algorithm": algo,
                "Mean Accuracy (%)": f"{m_acc:.2f}%",
                "Best Seed Run": f"{max_a:.2f}%",
                "Worst Seed Run": f"{min_a:.2f}%",
                "Accuracy Gap (Δ)": f"{rng_a:.3f}%",
                "Std Dev (ddof=1)": f"±{s_acc:.3f}%",
                "Avg Training Time": f"{avg_t:.2f}s",
                "Time Range (Min-Max)": f"{min_t:.1f}s - {max_t:.1f}s",
                "Seed Sensitivity": sens
            })
        
        summary_df = pd.DataFrame(summary_records)
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Full Algorithm Performance Matrix</div>', unsafe_allow_html=True)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # =======================================================================
        # 3. SEED-BY-SEED DEVIATION FROM MEAN (DIFFERENCE TABLE)
        # =======================================================================
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📈 Seed Fluctuation Table (Deviation from Model Mean)</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size: 13px; color: #94A3B8; margin-bottom: 12px;">'
            'Shows exactly how much each seed deviated (above or below) from that model\'s overall average accuracy.'
            '</div>',
            unsafe_allow_html=True
        )

        pivot_dev = pivot_acc.copy()
        for col in pivot_dev.columns:
            col_mean = pivot_dev[col].mean()
            pivot_dev[col] = pivot_dev[col] - col_mean

        def format_diff(val):
            sign = "+" if val >= 0 else ""
            return f"{sign}{val:.3f}%"

        def highlight_diff(s):
            return [
                'color: #38BDF8; font-weight: 600;' if v >= 0 else
                'color: #F87171; font-weight: 600;'
                for v in s
            ]

        styled_dev = pivot_dev.style.apply(highlight_diff, axis=0).format(format_diff)
        st.dataframe(styled_dev, use_container_width=True)
        st.caption("🔵 Blue = Positive deviation (above average) | 🔴 Red = Negative deviation (below average).")
        st.markdown('</div>', unsafe_allow_html=True)

        # =======================================================================
        # 4. COMPARATIVE VISUALIZATIONS
        # =======================================================================
        col_c1, col_c2 = st.columns([1.5, 1.2])
        
        with col_c1:
            fig_comp = go.Figure()
            palette = {"Random Forest": "#38BDF8", "Neural Network (MLP)": "#A78BFA", "Logistic Regression": "#34D399"}
            
            for algo in analysis_df["Algorithm"].unique():
                sub = analysis_df[analysis_df["Algorithm"] == algo].sort_values(by="Seed")
                color = palette.get(algo, "#FBBF24")
                fig_comp.add_trace(go.Scatter(
                    x=[f"Seed {int(s)}" for s in sub["Seed"]],
                    y=sub["Accuracy_Pct"],
                    mode="lines+markers",
                    name=algo,
                    line=dict(width=2.5, color=color),
                    marker=dict(size=8),
                    hovertemplate=f"<b>{algo}</b><br>%{{x}}: %{{y:.2f}}%<extra></extra>"
                ))

            fig_comp.update_layout(
                title=dict(text="<b>Accuracy Trajectory Across Seeds (All Models)</b>", font=dict(size=14, color="#FFFFFF")),
                xaxis=dict(title=dict(text="Random Seed", font=dict(color="#94A3B8")), gridcolor="#334155", tickfont=dict(color="#E2E8F0")),
                yaxis=dict(title=dict(text="Accuracy (%)", font=dict(color="#94A3B8")), gridcolor="#334155", tickfont=dict(color="#E2E8F0")),
                plot_bgcolor="#1E293B",
                paper_bgcolor="#1E293B",
                legend=dict(font=dict(color="#FFFFFF")),
                margin=dict(l=45, r=20, t=45, b=40)
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        with col_c2:
            fig_sens = go.Figure()
            sens_df = analysis_df.groupby("Algorithm")["Accuracy_Pct"].std(ddof=1).reset_index()
            fig_sens.add_trace(go.Bar(
                x=sens_df["Algorithm"],
                y=sens_df["Accuracy_Pct"],
                marker_color=["#38BDF8", "#A78BFA", "#34D399"][:len(sens_df)],
                hovertemplate="<b>%{x}</b><br>Std Dev: ±%{y:.3f}%<extra></extra>"
            ))
            fig_sens.update_layout(
                title=dict(text="<b>Seed Sensitivity Ranking (Std Dev %)</b>", font=dict(size=14, color="#FFFFFF")),
                xaxis=dict(title=dict(text="Algorithm", font=dict(color="#94A3B8")), gridcolor="#334155", tickfont=dict(color="#E2E8F0")),
                yaxis=dict(title=dict(text="Std Dev (%)", font=dict(color="#94A3B8")), gridcolor="#334155", tickfont=dict(color="#E2E8F0")),
                plot_bgcolor="#1E293B",
                paper_bgcolor="#1E293B",
                margin=dict(l=45, r=20, t=45, b=40)
            )
            st.plotly_chart(fig_sens, use_container_width=True)

        # =======================================================================
        # 5. EASY-TO-UNDERSTAND INSIGHTS & CONCLUSION (VIVA READY)
        # =======================================================================
        col_ins1, col_ins2 = st.columns(2)

        with col_ins1:
            st.markdown("""
            <div class="dashboard-card">
                <div class="card-title">💡 Key Insights (In Simple Words)</div>
                <div style="font-size: 13.5px; color: #CBD5E1; line-height: 1.6;">
                    <b>1. Logistic Regression was completely flat (0.00% change):</b><br/>
                    It got exactly 80.66% on all 5 seeds because linear math has only <i>one best answer</i>. Seed does not affect it.<br/><br/>
                    <b>2. Neural Network showed small shifts (0.16% change):</b><br/>
                    Different seeds started the weights at different spots, but it learned consistently.<br/><br/>
                    <b>3. Random Forest had the biggest jumps (0.53% change & 2.5x time swing):</b><br/>
                    Because it randomly picks rows and features for each tree, different seeds produce completely different trees and varying training speeds!
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_ins2:
            st.markdown("""
            <div class="dashboard-card">
                <div class="card-title">🎓 Final Conclusion (What to Tell the Examiner)</div>
                <div style="font-size: 13.5px; color: #CBD5E1; line-height: 1.6;">
                    <b>"Never trust a machine learning result from a single seed."</b><br/><br/>
                    In Random Forest, Seed 2 scored <b>94.60%</b> while Seed 4 scored <b>94.07%</b> on the exact same data. 
                    If someone reports only Seed 2, they are showing <b>seed luck</b>, not a better model.<br/><br/>
                    <b>Scientific Takeaway:</b> Always train models over at least 5 different random seeds and report the <b>average (mean) and standard deviation</b>.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # Case B: Individual Algorithm Focus (Single Model Drill-Down)
    # ---------------------------------------------------------------------------
    else:
        filtered_df = analysis_df if not has_multi_algo else analysis_df[analysis_df["Algorithm"] == selected_algo]
        algo_label = selected_algo if selected_algo else "Model"

        n_runs = len(filtered_df)
        mean_acc = float(filtered_df["Accuracy_Pct"].mean())
        std_acc = float(filtered_df["Accuracy_Pct"].std(ddof=1)) if n_runs > 1 else 0.0
        min_acc = float(filtered_df["Accuracy_Pct"].min())
        max_acc = float(filtered_df["Accuracy_Pct"].max())
        range_acc = float(max_acc - min_acc)

        st.caption(f"Analyzing **{algo_label}** across **{n_runs} experimental runs** — {st.session_state.data_source_label}")

        # 5 Key Metric Cards
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        with col_m1:
            st.markdown(f"""
            <div class="metric-stat-card">
                <div class="metric-stat-title">Mean Accuracy</div>
                <div class="metric-stat-number">{mean_acc:.2f}%</div>
                <div class="metric-stat-subtext">Average Performance</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div class="metric-stat-card">
                <div class="metric-stat-title">Std Deviation</div>
                <div class="metric-stat-number">±{std_acc:.3f}%</div>
                <div class="metric-stat-subtext">Spread (ddof=1)</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
            <div class="metric-stat-card">
                <div class="metric-stat-title">Lowest Run</div>
                <div class="metric-stat-number" style="color: #FBBF24;">{min_acc:.2f}%</div>
                <div class="metric-stat-subtext" style="color: #FBBF24;">Lowest Seed</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m4:
            st.markdown(f"""
            <div class="metric-stat-card">
                <div class="metric-stat-title">Highest Run</div>
                <div class="metric-stat-number" style="color: #34D399;">{max_acc:.2f}%</div>
                <div class="metric-stat-subtext" style="color: #34D399;">Highest Seed</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m5:
            st.markdown(f"""
            <div class="metric-stat-card">
                <div class="metric-stat-title">Seed Gap (Range)</div>
                <div class="metric-stat-number" style="color: #38BDF8;">{range_acc:.3f}%</div>
                <div class="metric-stat-subtext">Max - Min Variation</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # Individual Plotly Charts
        df_sorted = filtered_df.sort_values(by="Seed")
        col_c_left, col_c_right = st.columns([1.5, 1.2])

        with col_c_left:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                x=[f"Seed {int(s)}" for s in df_sorted["Seed"]],
                y=df_sorted["Accuracy_Pct"],
                mode="lines+markers",
                name="Accuracy (%)",
                line=dict(color="#38BDF8", width=3),
                marker=dict(size=10, color="#FFFFFF", line=dict(color="#0284C7", width=2)),
                hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.3f}%<extra></extra>"
            ))
            fig_acc.add_hline(
                y=mean_acc,
                line_dash="dash",
                line_color="#A78BFA",
                line_width=1.5,
                annotation_text=f"Mean: {mean_acc:.2f}%",
                annotation_position="bottom right",
                annotation_font=dict(color="#E2E8F0", size=11)
            )
            fig_acc.update_layout(
                title=dict(text=f"<b>{algo_label}: Accuracy Across Seeds</b>", font=dict(size=14, color="#FFFFFF"), x=0.01, y=0.96),
                xaxis=dict(title=dict(text="Random Seed", font=dict(color="#94A3B8")), gridcolor="#334155", tickfont=dict(color="#E2E8F0")),
                yaxis=dict(title=dict(text="Accuracy (%)", font=dict(color="#94A3B8")), gridcolor="#334155", tickfont=dict(color="#E2E8F0")),
                plot_bgcolor="#1E293B",
                paper_bgcolor="#1E293B",
                margin=dict(l=45, r=20, t=45, b=40),
                hoverlabel=dict(bgcolor="#0F172A", font_size=12, font_color="#FFFFFF")
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        with col_c_right:
            fig_time = go.Figure()
            fig_time.add_trace(go.Bar(
                x=[f"Seed {int(s)}" for s in df_sorted["Seed"]],
                y=df_sorted["Training Time"],
                marker_color="#818CF8",
                hovertemplate="<b>%{x}</b><br>Training Time: %{y:.2f}s<extra></extra>"
            ))
            fig_time.update_layout(
                title=dict(text=f"<b>{algo_label}: Training Time Across Runs</b>", font=dict(size=14, color="#FFFFFF"), x=0.01, y=0.96),
                xaxis=dict(title=dict(text="Random Seed", font=dict(color="#94A3B8")), gridcolor="#334155", tickfont=dict(color="#E2E8F0")),
                yaxis=dict(title=dict(text="Seconds", font=dict(color="#94A3B8")), gridcolor="#334155", tickfont=dict(color="#E2E8F0")),
                plot_bgcolor="#1E293B",
                paper_bgcolor="#1E293B",
                margin=dict(l=45, r=20, t=45, b=40),
                hoverlabel=dict(bgcolor="#0F172A", font_size=12, font_color="#FFFFFF")
            )
            st.plotly_chart(fig_time, use_container_width=True)

        # Detailed Run Table
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="card-title">📋 Run Table ({algo_label})</div>', unsafe_allow_html=True)

        table_df = filtered_df.copy().sort_values(by="Seed")
        table_df.insert(0, "Run #", range(1, len(table_df) + 1))
        
        display_cols = ["Run #", "Seed", "Accuracy_Pct", "Training Time"]
        display_tbl = table_df[display_cols].rename(columns={
            "Seed": "Random Seed",
            "Accuracy_Pct": "Accuracy (%)",
            "Training Time": "Training Time (s)"
        })

        def highlight_extrema_dark(s):
            is_max = s == s.max()
            is_min = s == s.min()
            return [
                'background-color: rgba(52, 211, 153, 0.25); color: #34D399; font-weight: 700;' if v_max else
                'background-color: rgba(251, 191, 36, 0.25); color: #FBBF24; font-weight: 700;' if v_min else ''
                for v_max, v_min in zip(is_max, is_min)
            ]

        styled_tbl = display_tbl.style.apply(highlight_extrema_dark, subset=["Accuracy (%)"]).format({
            "Random Seed": "{:.0f}",
            "Accuracy (%)": "{:.3f}%",
            "Training Time (s)": "{:.2f}s"
        })
        st.dataframe(styled_tbl, use_container_width=True, hide_index=True)
        st.caption("🟢 Green = Highest accuracy | 🟡 Yellow = Lowest accuracy")
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # CSV Download Button
    # ---------------------------------------------------------------------------
    col_dl, _ = st.columns([1.5, 3.5])
    with col_dl:
        export_cols = [c for c in ["Algorithm", "Seed", "Accuracy", "Training Time"] if c in analysis_df.columns]
        csv_data = analysis_df[export_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Full Results CSV",
            data=csv_data,
            file_name="seedlab_results_analyzed.csv",
            mime="text/csv",
            use_container_width=True
        )
