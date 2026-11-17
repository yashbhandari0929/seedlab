"""
SeedLab AI — View: Model Prediction & Interactive Digit Recognition.
Polished, responsive, scientifically authentic, and uncluttered.
"""

from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_drawable_canvas import st_canvas

from utils.model_loader import load_model_suite
from utils.image_preprocessing import preprocess_canvas_image
from utils.prediction import predict_digit_probabilities


def render_model_prediction():
    """Renders the clean, elegant Live Digit Prediction interface."""
    st.markdown('<div class="page-header-title">Model Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-header-subtitle">Draw a handwritten digit (0–9) and test your selected machine learning model live.</div>',
        unsafe_allow_html=True
    )

    # ---------------------------------------------------------------------------
    # Step 1: Ensure Model Suite is Loaded
    # ---------------------------------------------------------------------------
    if "model_suite" not in st.session_state:
        st.session_state["model_suite"] = load_model_suite()

    suite = st.session_state["model_suite"]
    meta = suite.get("metadata", {}) or {}
    models_meta = meta.get("models_evaluated", {}) if meta else {}

    # ---------------------------------------------------------------------------
    # Step 2: Determine Winning Model
    # ---------------------------------------------------------------------------
    selection_res = st.session_state.get("model_selection_results", None)

    if selection_res is not None:
        best_model_info = selection_res["best_model"]
        winning_model_name = best_model_info["name"]
        mean_f1 = best_model_info["mean_f1"]
        std_f1 = best_model_info["std_f1"]
        mean_acc = best_model_info["mean_accuracy"]
        rationale = best_model_info["reason"]
        source_note = "Selected dynamically from your uploaded experiment CSV."
    else:
        # Default benchmark winner if CSV hasn't been uploaded yet
        winning_model_name = "Random Forest"
        winner_test = models_meta.get(winning_model_name, {}).get("heldout_test_metrics", {})
        mean_f1 = winner_test.get("macro_f1", 0.9444)
        std_f1 = 0.00197
        mean_acc = winner_test.get("accuracy", 0.9448)
        rationale = "Achieved highest mean macro F1-score across 5 random seeds with minimal variance."
        source_note = "Using benchmark winner. Upload your CSV on the Experiment Dashboard to dynamically select from your own runs."

    winning_estimator = suite["models"].get(winning_model_name, None)
    winner_meta = models_meta.get(winning_model_name, {})
    is_calibrated = winner_meta.get("is_calibrated", True)

    # ---------------------------------------------------------------------------
    # SECTION A: Selected Model Information Card (4 Crisp Metric Cards)
    # ---------------------------------------------------------------------------
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🏆 Selected Best Model</div>', unsafe_allow_html=True)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.markdown(f"""
        <div class="metric-stat-card">
            <div class="metric-stat-title">Winning Model</div>
            <div class="metric-stat-number" style="color: #38BDF8; font-size: 20px;">{winning_model_name}</div>
            <div class="metric-stat-subtext" style="color: #34D399; font-weight: 600;">● Winner (Rank #1)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown(f"""
        <div class="metric-stat-card">
            <div class="metric-stat-title">Mean Macro F1</div>
            <div class="metric-stat-number">{mean_f1*100:.2f}%</div>
            <div class="metric-stat-subtext">Spread: ±{std_f1*100:.3f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m3:
        st.markdown(f"""
        <div class="metric-stat-card">
            <div class="metric-stat-title">Mean Accuracy</div>
            <div class="metric-stat-number">{mean_acc*100:.2f}%</div>
            <div class="metric-stat-subtext">Across Tested Seeds</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m4:
        cal_status_badge = "🟢 Calibrated" if is_calibrated else "🟡 Raw"
        cal_sub = "Platt Scaling (cv=3)" if is_calibrated else "Uncalibrated"
        st.markdown(f"""
        <div class="metric-stat-card">
            <div class="metric-stat-title">Probability Status</div>
            <div class="metric-stat-number" style="color: #34D399; font-size: 20px;">{cal_status_badge}</div>
            <div class="metric-stat-subtext">{cal_sub}</div>
        </div>
        """, unsafe_allow_html=True)

    # Concise rationale banner
    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid #334155; border-radius: 8px; padding: 10px 14px; margin-top: 14px; font-size: 13px; color: #CBD5E1; line-height: 1.4;">
        <span style="font-weight: 700; color: #38BDF8;">Selection Rationale: </span>{rationale}
        <span style="color: #94A3B8; font-size: 11.5px; display: block; margin-top: 4px;">ℹ️ {source_note}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if winning_estimator is None:
        st.error(f"Estimator file for '{winning_model_name}' could not be loaded from models/ directory.")
        return

    # ---------------------------------------------------------------------------
    # SECTION B & C: Drawing Canvas (Left) & Real-Time Prediction (Right)
    # ---------------------------------------------------------------------------
    col_canvas_col, col_results_col = st.columns([1.1, 1.4], gap="large")

    with col_canvas_col:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">✍️ Interactive Drawing Canvas</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size: 12.5px; color: #94A3B8; margin-bottom: 12px;">'
            'Draw a single digit <b>(0 to 9)</b> in white on the black 280×280 canvas below.'
            '</div>',
            unsafe_allow_html=True
        )

        if "canvas_version" not in st.session_state:
            st.session_state["canvas_version"] = 0

        canvas_key = f"canvas_mnist_{st.session_state['canvas_version']}"

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
    # Handle Prediction Logic
    # ---------------------------------------------------------------------------
    if "prediction_history" not in st.session_state:
        st.session_state["prediction_history"] = []

    if predict_clicked:
        if canvas_result is not None and canvas_result.image_data is not None:
            norm_28, flat_784, is_blank, _ = preprocess_canvas_image(canvas_result.image_data)

            if is_blank:
                st.warning("⚠️ Please draw a digit (0–9) on the canvas before predicting.")
            else:
                pred_result = predict_digit_probabilities(winning_estimator, flat_784, winner_meta)

                active_payload = {
                    "model_name": winning_model_name,
                    "predicted_digit": pred_result["predicted_digit"],
                    "probability": pred_result["predicted_class_probability"],
                    "prob_label": pred_result["probability_label"],
                    "is_calibrated": pred_result["is_calibrated"],
                    "cal_status": pred_result["calibration_status"],
                    "probabilities": pred_result["probabilities"],
                    "interpretation": pred_result["scientific_interpretation"],
                    "norm_28": norm_28,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                }
                st.session_state["active_prediction"] = active_payload

                history_entry = {
                    "Time": active_payload["timestamp"],
                    "Model": winning_model_name,
                    "Predicted Digit": active_payload["predicted_digit"],
                    "Probability": f"{active_payload['probability']*100:.2f}%"
                }
                st.session_state["prediction_history"].insert(0, history_entry)
                st.session_state["prediction_history"] = st.session_state["prediction_history"][:5]

    active_pred = st.session_state.get("active_prediction", None)

    # ---------------------------------------------------------------------------
    # Prediction Results Panel
    # ---------------------------------------------------------------------------
    with col_results_col:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🎯 Prediction Results</div>', unsafe_allow_html=True)

        if active_pred is None:
            st.markdown("""
            <div style="text-align: center; padding: 45px 20px; color: #94A3B8;">
                <div style="font-size: 38px; margin-bottom: 8px;">🖌️</div>
                <div style="font-weight: 600; font-size: 16px; color: #E2E8F0;">Ready for Drawing</div>
                <div style="font-size: 13px; margin-top: 4px;">Draw any digit on the left canvas and click <b>Predict Digit</b> to see live recognition.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            digit = active_pred["predicted_digit"]
            prob_pct = active_pred["probability"] * 100.0
            prob_label = active_pred["prob_label"]
            cal_status = active_pred["cal_status"]
            interpretation = active_pred["interpretation"]

            # Sleek High-Contrast Prediction Banner
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.95)); border: 2px solid #38BDF8; border-radius: 12px; padding: 18px 24px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; box-shadow: 0 4px 14px rgba(56, 189, 248, 0.25);">
                <div>
                    <div style="font-size: 11px; text-transform: uppercase; font-weight: 700; color: #94A3B8; letter-spacing: 0.08em;">PREDICTED DIGIT</div>
                    <div style="font-size: 56px; font-weight: 900; color: #FFFFFF; line-height: 1; margin-top: 4px;">{digit}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 11px; text-transform: uppercase; font-weight: 700; color: #94A3B8; letter-spacing: 0.08em;">{prob_label.upper()}</div>
                    <div style="font-size: 34px; font-weight: 800; color: #34D399; margin-top: 2px;">{prob_pct:.2f}%</div>
                    <div style="font-size: 11.5px; color: #38BDF8; font-weight: 600;">{cal_status}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Scientific Interpretation Callout
            st.markdown(f"""
            <div style="font-size: 12.5px; color: #CBD5E1; margin-bottom: 12px; padding: 6px 12px; background: rgba(15, 23, 42, 0.5); border-left: 3px solid #38BDF8; border-radius: 0 6px 6px 0;">
                <b>Model Assessment:</b> {interpretation}
            </div>
            """, unsafe_allow_html=True)

            # Centered 28x28 Preprocessed Tensor Preview + Probability Chart Side-by-Side
            col_tensor, col_chart = st.columns([1, 2.5])

            with col_tensor:
                upscaled = (active_pred["norm_28"] * 255.0).astype(np.uint8)
                st.image(upscaled, width=95, caption="28×28 Centered Input", clamp=True)

            with col_chart:
                probs = active_pred["probabilities"]
                digits = [str(d) for d in range(10)]
                colors = ["#38BDF8" if i == digit else "#334155" for i in range(10)]

                fig_prob = go.Figure()
                fig_prob.add_trace(go.Bar(
                    x=digits,
                    y=[p * 100 for p in probs],
                    marker_color=colors,
                    hovertemplate="<b>Digit %{x}</b><br>Probability: %{y:.2f}%<extra></extra>",
                    text=[f"{p*100:.1f}%" if p > 0.04 else "" for p in probs],
                    textposition="outside",
                    textfont=dict(color="#FFFFFF", size=9)
                ))

                fig_prob.update_layout(
                    title=dict(text="<b>0–9 Class Probabilities</b>", font=dict(size=13, color="#FFFFFF")),
                    xaxis=dict(gridcolor="#1E293B", tickfont=dict(color="#E2E8F0")),
                    yaxis=dict(gridcolor="#334155", tickfont=dict(color="#E2E8F0"), range=[0, max(100, max(probs)*100 + 15)]),
                    plot_bgcolor="#1E293B",
                    paper_bgcolor="#1E293B",
                    margin=dict(l=30, r=10, t=30, b=25),
                    height=180
                )
                st.plotly_chart(fig_prob, use_container_width=True)

            st.caption("ℹ️ Probabilities reflect estimated class likelihoods from model inference.")

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # Recent Prediction History Table (Compact & Clean)
    # ---------------------------------------------------------------------------
    history = st.session_state.get("prediction_history", [])
    if history:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        col_h1, col_h2 = st.columns([3.5, 1])
        with col_h1:
            st.markdown('<div class="card-title">📜 Recent Prediction History</div>', unsafe_allow_html=True)
        with col_h2:
            if st.button("Clear History", use_container_width=True):
                st.session_state["prediction_history"] = []
                st.rerun()

        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
