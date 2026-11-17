"""
SeedLab AI — View: Model Prediction & Interactive Digit Recognition.
Clean, precise, and scientifically authentic without UI clutter.
Zero in-app training; reads real probabilities directly from the loaded estimator.
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
    """Renders a streamlined, focused live digit prediction interface."""
    st.markdown('<div class="page-header-title">Model Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-header-subtitle">Draw a handwritten digit (0–9) and test your selected machine learning model.</div>',
        unsafe_allow_html=True
    )

    # ---------------------------------------------------------------------------
    # Check if Experiment Results CSV has been uploaded on Dashboard
    # ---------------------------------------------------------------------------
    selection_res = st.session_state.get("model_selection_results", None)

    if selection_res is None:
        st.markdown("""
        <div class="empty-upload-box">
            <div style="font-size: 40px; margin-bottom: 8px;">⏳</div>
            <div class="empty-upload-title">Awaiting Experiment Results</div>
            <div class="empty-upload-text">
                No experiment results are currently loaded in this session.<br/>
                Please navigate to the <b>Experiment Dashboard</b> on the left sidebar and upload your 
                <code>seedlab_results.csv</code> file. SeedLab AI will automatically identify the best model 
                and activate live digit prediction here.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    best_model_info = selection_res["best_model"]
    winning_model_name = best_model_info["name"]

    # ---------------------------------------------------------------------------
    # Load Model Suite (local models directory)
    # ---------------------------------------------------------------------------
    if "model_suite" not in st.session_state:
        st.session_state["model_suite"] = load_model_suite()

    suite = st.session_state["model_suite"]
    winning_estimator = suite["models"].get(winning_model_name, None)
    meta = suite.get("metadata", {}) or {}
    models_meta = meta.get("models_evaluated", {}) if meta else {}
    winner_meta = models_meta.get(winning_model_name, {})

    # ---------------------------------------------------------------------------
    # Clean, Compact Selected Model Information Card
    # ---------------------------------------------------------------------------
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    col_info1, col_info2, col_info3 = st.columns([1.5, 1.2, 1.3])

    with col_info1:
        st.markdown(f"""
        <div style="font-size: 11px; text-transform: uppercase; font-weight: 700; color: #94A3B8; letter-spacing: 0.05em;">Selected Model</div>
        <div style="font-size: 20px; font-weight: 800; color: #38BDF8; margin-top: 2px;">{winning_model_name}</div>
        <div style="font-size: 11.5px; color: #34D399; font-weight: 600; margin-top: 2px;">● Deterministic Winner (Rank #1)</div>
        """, unsafe_allow_html=True)

    with col_info2:
        m_f1_pct = best_model_info["mean_f1"] * 100
        m_acc_pct = best_model_info["mean_accuracy"] * 100
        st.markdown(f"""
        <div style="font-size: 11px; text-transform: uppercase; font-weight: 700; color: #94A3B8; letter-spacing: 0.05em;">Performance Metrics</div>
        <div style="font-size: 15px; font-weight: 700; color: #FFFFFF; margin-top: 2px;">F1: {m_f1_pct:.2f}% | Acc: {m_acc_pct:.2f}%</div>
        <div style="font-size: 11.5px; color: #94A3B8; margin-top: 2px;">Variability: ±{best_model_info['std_f1']*100:.3f}%</div>
        """, unsafe_allow_html=True)

    with col_info3:
        is_cal = winner_meta.get("is_calibrated", False)
        cal_label = "🟢 Calibrated Probabilities" if is_cal else "🟡 Raw Probabilities"
        status_text = "Ready for live inference" if winning_estimator is not None else "Artifact missing"
        st.markdown(f"""
        <div style="font-size: 11px; text-transform: uppercase; font-weight: 700; color: #94A3B8; letter-spacing: 0.05em;">Status</div>
        <div style="font-size: 15px; font-weight: 700; color: #34D399; margin-top: 2px;">{cal_label}</div>
        <div style="font-size: 11.5px; color: #94A3B8; margin-top: 2px;">{status_text}</div>
        """, unsafe_allow_html=True)

    # Concise rationale
    st.markdown(f"""
    <div style="font-size: 12.5px; color: #CBD5E1; margin-top: 10px; padding-top: 8px; border-top: 1px solid #334155; line-height: 1.4;">
        <span style="color: #38BDF8; font-weight: 600;">Selection Reason:</span> {best_model_info['reason']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if winning_estimator is None:
        st.error(f"Model artifact for '{winning_model_name}' was not found in the models directory.")
        return

    # ---------------------------------------------------------------------------
    # Interactive Workspace: Drawing Canvas (Left) + Prediction Results (Right)
    # ---------------------------------------------------------------------------
    col_canvas, col_results = st.columns([1.1, 1.4], gap="large")

    with col_canvas:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">✍️ Draw a Digit</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size: 12.5px; color: #94A3B8; margin-bottom: 10px;">'
            'Draw a clear single digit <b>(0 to 9)</b> in the center of the canvas.'
            '</div>',
            unsafe_allow_html=True
        )

        if "canvas_version" not in st.session_state:
            st.session_state["canvas_version"] = 0

        canvas_key = f"canvas_v_{st.session_state['canvas_version']}"

        # 280x280 Drawing Canvas
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

        btn_col1, btn_col2 = st.columns([1.5, 1])
        predict_clicked = False
        with btn_col1:
            predict_clicked = st.button("🚀 Predict Digit", type="primary", use_container_width=True)
        with btn_col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state["canvas_version"] += 1
                st.session_state["active_prediction"] = None
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # Inference Execution
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
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                }
                st.session_state["active_prediction"] = active_payload

                # Keep last 5 runs in history
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
    # Prediction Results Display
    # ---------------------------------------------------------------------------
    with col_results:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🎯 Prediction Results</div>', unsafe_allow_html=True)

        if active_pred is None:
            st.markdown("""
            <div style="text-align: center; padding: 45px 20px; color: #94A3B8;">
                <div style="font-size: 36px; margin-bottom: 8px;">🖌️</div>
                <div style="font-weight: 600; font-size: 15px; color: #E2E8F0;">Canvas Ready</div>
                <div style="font-size: 12.5px; margin-top: 4px;">Draw a digit on the left and click <b>Predict Digit</b> to see live recognition.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            digit = active_pred["predicted_digit"]
            prob_pct = active_pred["probability"] * 100.0
            prob_label = active_pred["prob_label"]
            cal_status = active_pred["cal_status"]
            interpretation = active_pred["interpretation"]

            # Prominent Clean Prediction Banner
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.95)); border: 2px solid #38BDF8; border-radius: 12px; padding: 18px 24px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; box-shadow: 0 4px 14px rgba(56, 189, 248, 0.2);">
                <div>
                    <div style="font-size: 11px; text-transform: uppercase; font-weight: 700; color: #94A3B8; letter-spacing: 0.08em;">PREDICTED DIGIT</div>
                    <div style="font-size: 52px; font-weight: 900; color: #FFFFFF; line-height: 1; margin-top: 4px;">{digit}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 11px; text-transform: uppercase; font-weight: 700; color: #94A3B8; letter-spacing: 0.08em;">{prob_label.upper()}</div>
                    <div style="font-size: 34px; font-weight: 800; color: #34D399; margin-top: 2px;">{prob_pct:.2f}%</div>
                    <div style="font-size: 11.5px; color: #38BDF8; font-weight: 600;">{cal_status}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Scientific Interpretation Note
            st.markdown(f"""
            <div style="font-size: 12px; color: #CBD5E1; margin-bottom: 12px; padding: 6px 10px; background: rgba(15, 23, 42, 0.4); border-left: 3px solid #38BDF8; border-radius: 0 4px 4px 0;">
                <b>Model Assessment:</b> {interpretation}
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
                hovertemplate="<b>Digit %{x}</b><br>Probability: %{y:.2f}%<extra></extra>",
                text=[f"{p*100:.1f}%" if p > 0.04 else "" for p in probs],
                textposition="outside",
                textfont=dict(color="#FFFFFF", size=10)
            ))

            fig_prob.update_layout(
                title=dict(text="<b>Probability Distribution Across Classes (0 to 9)</b>", font=dict(size=13, color="#FFFFFF")),
                xaxis=dict(title=dict(text="Digit Class", font=dict(color="#94A3B8")), gridcolor="#1E293B", tickfont=dict(color="#E2E8F0")),
                yaxis=dict(title=dict(text="Probability (%)", font=dict(color="#94A3B8")), gridcolor="#334155", tickfont=dict(color="#E2E8F0"), range=[0, max(100, max(probs)*100 + 15)]),
                plot_bgcolor="#1E293B",
                paper_bgcolor="#1E293B",
                margin=dict(l=40, r=20, t=35, b=30),
                height=220
            )
            st.plotly_chart(fig_prob, use_container_width=True)

            st.caption("ℹ️ Probabilities represent the model's estimated class likelihoods, not guaranteed ground truth.")

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # Recent Prediction History (Last 5 Runs)
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
