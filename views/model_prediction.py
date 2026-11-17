"""
SeedLab AI — View: Model Prediction & Interactive Digit Recognition.
Handles automatic best model display, drawing canvas, preprocessed previews,
probability distribution, 3-model comparison, and prediction history.
"""

from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_drawable_canvas import st_canvas

from utils.model_selection import evaluate_and_select_best_model
from utils.model_loader import load_model_suite, CANONICAL_FILENAMES
from utils.image_preprocessing import preprocess_canvas_image
from utils.prediction import predict_digit_probabilities, predict_across_all_models


def render_model_prediction():
    """Renders the Live Digit Prediction and Best Model Evaluation interface."""
    st.markdown('<div class="page-header-title">Model Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-header-subtitle">Draw a handwritten digit and let your selected machine learning model recognize it.</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="page-header-desc">'
        'This interface uses the winning estimator selected from your experimental seed analysis. '
        'Draw any single digit (0–9) on the canvas below to run live inference through our standardized MNIST preprocessing pipeline.'
        '</div>',
        unsafe_allow_html=True
    )

    # ---------------------------------------------------------------------------
    # Step 1: Ensure Experiment Results & Best Model Selection
    # ---------------------------------------------------------------------------
    selection_res = st.session_state.get("model_selection_results", None)

    # If user hasn't uploaded CSV yet, check if data/seedlab_results.csv is available for one-click load
    if selection_res is None:
        local_csv = Path(__file__).resolve().parent.parent / "data" / "seedlab_results.csv"
        st.warning("⚠️ No experiment CSV results loaded yet.")
        col_w1, col_w2 = st.columns([1.5, 2.5])
        with col_w1:
            if local_csv.exists() and st.button("⚡ Load Benchmark Experiment Results", use_container_width=True):
                bench_df = pd.read_csv(local_csv)
                st.session_state["experiment_df"] = bench_df
                st.session_state["data_source_label"] = "Loaded: data/seedlab_results.csv (Benchmark)"
                selection_res = evaluate_and_select_best_model(bench_df)
                st.session_state["model_selection_results"] = selection_res
                st.rerun()
        with col_w2:
            st.caption("Or navigate to **Experiment Dashboard** to upload your custom experiment CSV.")
        
        if selection_res is None:
            return

    best_model_info = selection_res["best_model"]
    summary_df = selection_res["summary_df"]
    winning_model_name = best_model_info["name"]

    # ---------------------------------------------------------------------------
    # Step 2: Load Model Suite (local directory or uploaded ZIP fallback)
    # ---------------------------------------------------------------------------
    if "model_suite" not in st.session_state:
        st.session_state["model_suite"] = load_model_suite()

    suite = st.session_state["model_suite"]
    winning_estimator = suite["models"].get(winning_model_name, None)

    # ---------------------------------------------------------------------------
    # SECTION A: Selected Model Information & Comparison Matrix
    # ---------------------------------------------------------------------------
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🏆 Selected Best Model: Dynamic Selection</div>', unsafe_allow_html=True)

    col_card1, col_card2, col_card3, col_card4 = st.columns([1.8, 1.2, 1.2, 1.4])

    with col_card1:
        st.markdown(f"""
        <div style="font-size: 13px; color: #94A3B8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Winning Model</div>
        <div style="font-size: 22px; font-weight: 800; color: #38BDF8; margin-top: 2px;">{winning_model_name}</div>
        <div style="font-size: 12px; color: #34D399; font-weight: 600; margin-top: 4px;">● Deterministic Winner (Rank #1)</div>
        """, unsafe_allow_html=True)

    with col_card2:
        m_f1_pct = best_model_info["mean_f1"] * 100
        std_f1_pct = best_model_info["std_f1"] * 100
        metric_label = "Mean Macro F1" if not best_model_info["used_fallback"] else "Mean Accuracy (Fallback)"
        st.markdown(f"""
        <div style="font-size: 12px; color: #94A3B8; font-weight: 600;">{metric_label}</div>
        <div style="font-size: 20px; font-weight: 800; color: #FFFFFF; margin-top: 2px;">{m_f1_pct:.2f}%</div>
        <div style="font-size: 11px; color: #94A3B8; margin-top: 2px;">Std Dev: ±{std_f1_pct:.3f}%</div>
        """, unsafe_allow_html=True)

    with col_card3:
        m_acc_pct = best_model_info["mean_accuracy"] * 100
        std_acc_pct = best_model_info["std_accuracy"] * 100
        st.markdown(f"""
        <div style="font-size: 12px; color: #94A3B8; font-weight: 600;">Mean Test Accuracy</div>
        <div style="font-size: 20px; font-weight: 800; color: #FFFFFF; margin-top: 2px;">{m_acc_pct:.2f}%</div>
        <div style="font-size: 11px; color: #94A3B8; margin-top: 2px;">Std Dev: ±{std_acc_pct:.3f}%</div>
        """, unsafe_allow_html=True)

    with col_card4:
        artifact_status = "🟢 Loaded & Ready" if winning_estimator is not None else "🔴 Artifact Missing"
        status_color = "#34D399" if winning_estimator is not None else "#F87171"
        st.markdown(f"""
        <div style="font-size: 12px; color: #94A3B8; font-weight: 600;">Model Artifact Status</div>
        <div style="font-size: 14px; font-weight: 700; color: {status_color}; margin-top: 4px;">{artifact_status}</div>
        <div style="font-size: 11px; color: #94A3B8; margin-top: 2px;">Avg Train Time: {best_model_info['mean_training_time']:.2f}s</div>
        """, unsafe_allow_html=True)

    # Selection reason box
    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid #334155; border-radius: 8px; padding: 12px 16px; margin-top: 14px; font-size: 13.5px; color: #E2E8F0; line-height: 1.5;">
        <span style="font-weight: 700; color: #38BDF8;">Selection Rationale: </span>
        {best_model_info['reason']}
    </div>
    """, unsafe_allow_html=True)

    # Compact Comparison Table of All 3 Models
    with st.expander("📊 View Compact Model Ranking & Comparison", expanded=False):
        comp_records = []
        for idx, row in summary_df.iterrows():
            is_winner = row["Algorithm"] == winning_model_name
            comp_records.append({
                "Rank": f"#{idx+1}" + (" 🏆" if is_winner else ""),
                "Model": row["Algorithm"],
                "Mean Macro F1": f"{row['Mean_F1']*100:.2f}%",
                "F1 Std Dev (±)": f"±{row['Std_F1']*100:.3f}%",
                "Mean Accuracy": f"{row['Mean_Accuracy']*100:.2f}%",
                "Accuracy Std Dev (±)": f"±{row['Std_Accuracy']*100:.3f}%",
                "Avg Latency": f"{row['Mean_Time']:.2f}s",
                "Selection Status": "Selected Winner" if is_winner else "Contender"
            })
        st.dataframe(pd.DataFrame(comp_records), use_container_width=True, hide_index=True)

    # Fallback ZIP Uploader if model artifact is missing
    if winning_estimator is None:
        st.markdown('<hr style="border: 0; border-top: 1px solid #334155; margin: 12px 0;" />', unsafe_allow_html=True)
        st.warning(f"⚠️ Model artifact for '{winning_model_name}' was not found in the local models/ directory.")
        st.markdown(
            "Upload your **seedlab_model_artifacts.zip** exported from Google Colab to load the fitted estimator:"
        )
        up_zip = st.file_uploader(
            "Upload Model Artifacts ZIP",
            type=["zip"],
            key="zip_uploader",
            help="Archive should contain random_forest.joblib, mlp.joblib, logistic_regression.joblib, metadata.json"
        )
        if up_zip is not None:
            new_suite = load_model_suite(uploaded_zip_file=up_zip)
            st.session_state["model_suite"] = new_suite
            st.success("Artifacts archive loaded successfully!")
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    if winning_estimator is None:
        st.error("Cannot proceed with digit recognition until the model artifact is loaded.")
        return

    # ---------------------------------------------------------------------------
    # SECTION B & C: Drawing Canvas & Real-Time Prediction
    # ---------------------------------------------------------------------------
    col_canvas_panel, col_results_panel = st.columns([1.1, 1.4], gap="large")

    with col_canvas_panel:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">✍️ Interactive Drawing Canvas</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size: 13px; color: #94A3B8; margin-bottom: 12px;">'
            'Draw a clear single digit <b>(0 to 9)</b> in white on the black 280×280 canvas.'
            '</div>',
            unsafe_allow_html=True
        )

        # Canvas dynamic key to allow one-click reset
        if "canvas_version" not in st.session_state:
            st.session_state["canvas_version"] = 0

        canvas_key = f"mnist_canvas_{st.session_state['canvas_version']}"

        # 280x280 Canvas (Black background, white stroke)
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=20,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key=canvas_key,
            display_toolbar=False
        )

        btn_c1, btn_c2 = st.columns([1.5, 1])
        predict_clicked = False
        with btn_c1:
            predict_clicked = st.button("🚀 Predict Digit", type="primary", use_container_width=True)
        with btn_c2:
            if st.button("🗑️ Clear Canvas", use_container_width=True):
                st.session_state["canvas_version"] += 1
                st.session_state["active_prediction"] = None
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # Process Drawing & Inference
    # ---------------------------------------------------------------------------
    if "prediction_history" not in st.session_state:
        st.session_state["prediction_history"] = []

    if predict_clicked:
        if canvas_result is not None and canvas_result.image_data is not None:
            # Preprocess canvas RGBA image
            norm_28, flat_784, is_blank, meta = preprocess_canvas_image(canvas_result.image_data)
            
            if is_blank:
                st.warning("⚠️ Please draw a digit (0–9) on the canvas before predicting.")
            else:
                # Infer with Winning Model
                pred_result = predict_digit_probabilities(winning_estimator, flat_784)
                
                # Infer with all 3 models for comparison
                all_models_pred = predict_across_all_models(suite["models"], flat_784)

                active_payload = {
                    "model_name": winning_model_name,
                    "predicted_digit": pred_result["predicted_digit"],
                    "confidence": pred_result["confidence"],
                    "probabilities": pred_result["probabilities"],
                    "norm_28": norm_28,
                    "all_models": all_models_pred,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                }
                st.session_state["active_prediction"] = active_payload

                # Add to history (keep top 5 latest)
                history_entry = {
                    "Time": active_payload["timestamp"],
                    "Model": winning_model_name,
                    "Digit": active_payload["predicted_digit"],
                    "Confidence": f"{active_payload['confidence']*100:.1f}%"
                }
                st.session_state["prediction_history"].insert(0, history_entry)
                st.session_state["prediction_history"] = st.session_state["prediction_history"][:5]

    active_pred = st.session_state.get("active_prediction", None)

    with col_results_panel:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🎯 Prediction Results</div>', unsafe_allow_html=True)

        if active_pred is None:
            st.markdown("""
            <div style="text-align: center; padding: 40px 20px; color: #94A3B8;">
                <div style="font-size: 38px; margin-bottom: 8px;">🖌️</div>
                <div style="font-weight: 600; font-size: 16px; color: #E2E8F0;">No Prediction Yet</div>
                <div style="font-size: 13px; margin-top: 4px;">Draw a digit on the left canvas and click <b>Predict Digit</b> to see live recognition.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            digit = active_pred["predicted_digit"]
            conf_pct = active_pred["confidence"] * 100

            # Prominent Predicted Digit Banner
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.95)); border: 2px solid #38BDF8; border-radius: 12px; padding: 18px 24px; display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; box-shadow: 0 4px 14px rgba(56, 189, 248, 0.25);">
                <div>
                    <div style="font-size: 12px; text-transform: uppercase; font-weight: 700; color: #94A3B8; letter-spacing: 0.05em;">Recognized Digit</div>
                    <div style="font-size: 52px; font-weight: 900; color: #FFFFFF; line-height: 1; margin-top: 4px;">{digit}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 12px; font-weight: 600; color: #94A3B8;">Confidence Score</div>
                    <div style="font-size: 32px; font-weight: 800; color: #34D399; margin-top: 2px;">{conf_pct:.1f}%</div>
                    <div style="font-size: 11px; color: #38BDF8; font-weight: 600;">{winning_model_name}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Preprocessing Pipeline Preview (Side-by-Side: 28x28 Preprocessed Array)
            with st.expander("🔬 View MNIST Preprocessed 28×28 Tensor", expanded=False):
                col_pv1, col_pv2 = st.columns([1, 2])
                with col_pv1:
                    # Upscale 28x28 normalized array for clear crisp display
                    upscaled = (active_pred["norm_28"] * 255.0).astype(np.uint8)
                    st.image(upscaled, width=140, caption="28×28 Centered MNIST Input", clamp=True)
                with col_pv2:
                    st.markdown("""
                    <div style="font-size: 12.5px; color: #94A3B8; line-height: 1.5;">
                        <b>Preprocessing Pipeline Steps Applied:</b><br/>
                        1. Grayscale conversion & bounding box extraction.<br/>
                        2. Aspect-ratio preserving resize into 20×20 bounding box.<br/>
                        3. Center-of-mass shift onto 28×28 matrix.<br/>
                        4. Normalization to [0.0, 1.0] float32 feature tensor (1×784).
                    </div>
                    """, unsafe_allow_html=True)

            # 0–9 Probability Distribution Bar Chart
            probs = active_pred["probabilities"]
            digits = [str(d) for d in range(10)]
            colors = ["#38BDF8" if i == digit else "#334155" for i in range(10)]

            fig_prob = go.Figure()
            fig_prob.add_trace(go.Bar(
                x=digits,
                y=[p * 100 for p in probs],
                marker_color=colors,
                hovertemplate="<b>Digit %{x}</b><br>Probability: %{y:.1f}%<extra></extra>",
                text=[f"{p*100:.1f}%" if p > 0.05 else "" for p in probs],
                textposition="outside",
                textfont=dict(color="#FFFFFF", size=10)
            ))

            fig_prob.update_layout(
                title=dict(text="<b>Probability Distribution (0 to 9)</b>", font=dict(size=14, color="#FFFFFF")),
                xaxis=dict(title=dict(text="Digit Class", font=dict(color="#94A3B8")), gridcolor="#1E293B", tickfont=dict(color="#E2E8F0")),
                yaxis=dict(title=dict(text="Probability (%)", font=dict(color="#94A3B8")), gridcolor="#334155", tickfont=dict(color="#E2E8F0"), range=[0, 105]),
                plot_bgcolor="#1E293B",
                paper_bgcolor="#1E293B",
                margin=dict(l=40, r=20, t=40, b=35),
                height=240
            )
            st.plotly_chart(fig_prob, use_container_width=True)

            # Collapsible 3-Model Cross-Comparison
            with st.expander("⚖️ Compare Prediction Across All Three Models", expanded=False):
                all_res = active_pred["all_models"]
                rows = []
                for mod_name, res in all_res.items():
                    d_pred = res["predicted_digit"]
                    c_pct = f"{res['confidence']*100:.1f}%" if res["confidence"] is not None else "N/A"
                    is_winner_flag = " (Selected Winner)" if mod_name == winning_model_name else ""
                    rows.append({
                        "Model": mod_name + is_winner_flag,
                        "Predicted Digit": str(d_pred) if d_pred is not None else "Err",
                        "Confidence": c_pct,
                        "Status": "Agreement ✅" if d_pred == digit else "Disagreement ⚠️"
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # SECTION D: Prediction History Table
    # ---------------------------------------------------------------------------
    history = st.session_state.get("prediction_history", [])
    if history:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        col_hist_h1, col_hist_h2 = st.columns([3, 1])
        with col_hist_h1:
            st.markdown('<div class="card-title">📜 Prediction History (Last 5 Runs)</div>', unsafe_allow_html=True)
        with col_hist_h2:
            if st.button("Clear History", use_container_width=True):
                st.session_state["prediction_history"] = []
                st.rerun()

        hist_df = pd.DataFrame(history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
