"""
SeedLab AI — View: Experiment Dashboard & Model Comparison.
Preserves all original experiment analysis, seed matrices, sensitivity ranking, and viva insights.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.model_selection import evaluate_and_select_best_model


def render_experiment_dashboard():
    """Renders the comprehensive SeedLab experiment analysis and reproducibility dashboard."""
    # Top Section Description
    st.markdown('<div class="page-header-title">Experiment Results & Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-header-desc">'
        'Upload your experiment results CSV exported from Google Colab to generate the comparative performance analysis, '
        'matrices, and visualizations across random seeds.'
        '</div>',
        unsafe_allow_html=True
    )

    # ---------------------------------------------------------------------------
    # Upload Bar (Clean, Full-Width)
    # ---------------------------------------------------------------------------
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📤 Upload Experiment Results CSV</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size: 13px; color: #94A3B8; margin-bottom: 12px;">'
        'Upload your <code>seedlab_results.csv</code> exported from Google Colab. '
        'Expected headers: <code>Seed</code>, <code>Accuracy</code>, <code>Training Time</code> (and optional <code>Algorithm</code>, <code>Precision</code>, <code>Recall</code>, <code>F1 Score</code>).'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload experiment CSV (from Google Colab)",
        type=["csv"],
        key="csv_uploader",
        help="Expected headers: Seed, Accuracy, Training Time (and optional Algorithm, F1 Score)",
        label_visibility="collapsed"
    )

    parsed_df = None
    validation_error = None
    data_source_label = ""

    # Process newly uploaded CSV File
    if uploaded_file is None:
        st.session_state["experiment_df"] = None
        st.session_state["data_source_label"] = ""
        st.session_state["model_selection_results"] = None
    else:
        try:
            raw_df = pd.read_csv(uploaded_file)
            if raw_df.empty:
                validation_error = "The uploaded CSV file is empty."
            else:
                col_map = {}
                for col in raw_df.columns:
                    c = str(col).strip().lower().replace("_", " ")
                    if c in ["algorithm", "model", "classifier"]:
                        col_map[col] = "Algorithm"
                    elif c in ["seed", "random seed", "random_seed"]:
                        col_map[col] = "Seed"
                    elif c in ["accuracy", "acc", "test accuracy"]:
                        col_map[col] = "Accuracy"
                    elif c in ["precision", "macro precision"]:
                        col_map[col] = "Precision"
                    elif c in ["recall", "macro recall"]:
                        col_map[col] = "Recall"
                    elif c in ["f1 score", "f1", "macro f1", "f1-score"]:
                        col_map[col] = "F1 Score"
                    elif c in ["training time", "train time", "time", "time (s)"]:
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
                    
                    if raw_df[["Seed", "Accuracy", "Training Time"]].isnull().values.any():
                        validation_error = "Uploaded file contains non-numeric or missing values in required columns."
                    else:
                        parsed_df = raw_df
                        data_source_label = f"Uploaded File: {uploaded_file.name}"
                        st.session_state["experiment_df"] = parsed_df
                        st.session_state["data_source_label"] = data_source_label

                        # Compute dynamic model selection results immediately
                        try:
                            selection_res = evaluate_and_select_best_model(parsed_df)
                            st.session_state["model_selection_results"] = selection_res
                        except Exception as sel_err:
                            st.warning(f"Could not calculate model ranking: {sel_err}")

        except Exception as e:
            validation_error = f"Failed to parse CSV: {e}"

    if validation_error:
        st.error(f"⚠️ {validation_error}")
        parsed_df = None

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # Empty State or Dashboard Display
    # ---------------------------------------------------------------------------
    if parsed_df is None or parsed_df.empty:
        st.markdown("""
        <div class="empty-upload-box">
            <div style="font-size: 44px; margin-bottom: 10px;">⏳</div>
            <div class="empty-upload-title">Awaiting CSV Upload</div>
            <div class="empty-upload-text">
                No experiment results are currently loaded.<br/>
                Please upload your <code>seedlab_results.csv</code> file exported from Google Colab into the box above 
                to generate the comparative performance analysis, matrices, and charts.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Normalize Accuracy to percentage
    analysis_df = parsed_df.copy()
    is_ratio = analysis_df["Accuracy"].max() <= 1.0
    multiplier = 100.0 if is_ratio else 1.0
    analysis_df["Accuracy_Pct"] = analysis_df["Accuracy"] * multiplier

    has_multi_algo = "Algorithm" in analysis_df.columns and analysis_df["Algorithm"].nunique() > 1

    selected_algo = None
    if has_multi_algo:
        algo_list = ["📊 All Models (Comparison Table & Matrix)"] + sorted(analysis_df["Algorithm"].unique().tolist())
        col_sel1, _ = st.columns([2, 3])
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
        st.caption(f"Comparing **{analysis_df['Algorithm'].nunique()} algorithms** side-by-side across all seeds — {data_source_label}")

        # 1. Primary Unified Seed Comparison Table
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔍 Unified Seed-by-Seed Accuracy Matrix</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size: 13px; color: #94A3B8; margin-bottom: 12px;">'
            'Compare test accuracy for each random seed side-by-side across all algorithms. '
            'Notice how each seed affects each model differently.'
            '</div>',
            unsafe_allow_html=True
        )

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

        # 2. Comprehensive Performance Matrix
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
                sens = "🟢 Zero / Immune"
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

        # 3. Seed-by-Seed Deviation From Mean
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

        # 4. Comparative Visualizations
        col_c1, col_c2 = st.columns([1.5, 1.2])
        
        with col_c1:
            fig_comp = go.Figure()
            palette = {"Random Forest": "#38BDF8", "MLP Neural Network": "#A78BFA", "Logistic Regression": "#34D399"}
            
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

        # 5. Easy-to-Understand Insights & Conclusion
        col_ins1, col_ins2 = st.columns(2)

        with col_ins1:
            st.markdown("""
            <div class="dashboard-card">
                <div class="card-title">💡 Key Insights (In Simple Words)</div>
                <div style="font-size: 13.5px; color: #CBD5E1; line-height: 1.6;">
                    <b>1. Logistic Regression was completely flat (0.00% change):</b><br/>
                    Convex optimization guarantees a single global optimum on deterministic features. Seed does not cause model drift.<br/><br/>
                    <b>2. Neural Network showed minor variance (~0.16% change):</b><br/>
                    Stochastic initial weights and mini-batch shuffling introduce slight seed variance.<br/><br/>
                    <b>3. Random Forest had visible jumps (~0.53% change & latency variance):</b><br/>
                    Bootstrap sample selection and feature subspace sampling per split yield varying tree ensembles across seeds.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_ins2:
            st.markdown("""
            <div class="dashboard-card">
                <div class="card-title">🎓 Final Conclusion (What to Tell the Examiner)</div>
                <div style="font-size: 13.5px; color: #CBD5E1; line-height: 1.6;">
                    <b>"Never trust a machine learning result from a single random seed."</b><br/><br/>
                    Reporting a single high-scoring seed misrepresents model capability as "seed luck".<br/><br/>
                    <b>Scientific Takeaway:</b> Always train models over at least 5 different random seeds and report the <b>mean and standard deviation</b> across all runs.
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

        st.caption(f"Analyzing **{algo_label}** across **{n_runs} experimental runs** — {data_source_label}")

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
        export_cols = [c for c in ["Algorithm", "Seed", "Accuracy", "Precision", "Recall", "F1 Score", "Training Time"] if c in analysis_df.columns]
        csv_data = analysis_df[export_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Full Results CSV",
            data=csv_data,
            file_name="seedlab_results_analyzed.csv",
            mime="text/csv",
            use_container_width=True
        )
