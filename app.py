"""
SeedLab AI — Machine Learning Reproducibility & Live Digit Prediction Platform.
Main Application Controller & Vertical Sidebar Routing.
"""

import sys
from pathlib import Path
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from views.experiment_dashboard import render_experiment_dashboard
from views.model_prediction import render_model_prediction

# ---------------------------------------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SeedLab | ML Reproducibility & Prediction",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Custom Dark CSS
css_path = APP_DIR / "assets" / "style.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Reactive Session State Synchronization
# ---------------------------------------------------------------------------
uploader_file = st.session_state.get("csv_uploader", None)
if uploader_file is not None and st.session_state.get("experiment_df", None) is None:
    try:
        import pandas as pd
        from utils.model_selection import evaluate_and_select_best_model
        raw_df = pd.read_csv(uploader_file)
        if not raw_df.empty:
            st.session_state["experiment_df"] = raw_df
            st.session_state["data_source_label"] = uploader_file.name
            st.session_state["model_selection_results"] = evaluate_and_select_best_model(raw_df)
    except Exception:
        pass
elif uploader_file is None and st.session_state.get("experiment_df", None) is not None:
    st.session_state["experiment_df"] = None
    st.session_state["data_source_label"] = ""
    st.session_state["model_selection_results"] = None

# ---------------------------------------------------------------------------
# Left Sidebar Navigation (EXACTLY TWO OPTIONS)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand-group">
        <div class="sidebar-brand-icon">SL</div>
        <div>
            <div class="sidebar-brand-title">SeedLab AI</div>
            <div class="sidebar-brand-subtitle">Reproducibility & Live Inference</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-nav-header">Navigation Menu</div>', unsafe_allow_html=True)

    nav_options = [
        "Experiment Dashboard",
        "Model Prediction"
    ]

    selected_nav = st.radio(
        "Navigation",
        options=nav_options,
        index=0,
        label_visibility="collapsed",
        key="main_navigation"
    )

    st.markdown('<hr style="border: 0; border-top: 1px solid #1E293B; margin: 24px 0 16px 0;" />', unsafe_allow_html=True)

    # Sidebar Status Indicator
    df_loaded = st.session_state.get("experiment_df", None) is not None
    selection_res = st.session_state.get("model_selection_results", None)
    
    if df_loaded and selection_res is not None:
        winning_name = selection_res["best_model"]["name"]
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 8px; padding: 12px 14px;">
            <div style="font-size: 11px; color: #94A3B8; text-transform: uppercase; font-weight: 700;">Active Pipeline</div>
            <div style="font-size: 13.5px; font-weight: 700; color: #34D399; margin-top: 2px;">● Results Loaded</div>
            <div style="font-size: 12px; color: #38BDF8; font-weight: 600; margin-top: 6px;">Winner: {winning_name}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 8px; padding: 12px 14px;">
            <div style="font-size: 11px; color: #94A3B8; text-transform: uppercase; font-weight: 700;">Status</div>
            <div style="font-size: 13.5px; font-weight: 700; color: #F59E0B; margin-top: 2px;">● Awaiting CSV Upload</div>
            <div style="font-size: 11.5px; color: #94A3B8; margin-top: 4px;">Upload results on Dashboard</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Top Header & Reactive Status Bar
# ---------------------------------------------------------------------------
col_nav_brand, col_nav_meta = st.columns([2.8, 1.4])

with col_nav_brand:
    st.markdown("""
    <div class="brand-title-group">
        <div class="brand-icon-box">SL</div>
        <div>
            <div class="brand-title-text">SeedLab</div>
            <div class="brand-subtitle-text">Machine Learning Reproducibility & Prediction Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_nav_meta:
    if st.session_state.get("experiment_df", None) is not None:
        source_name = st.session_state.get("data_source_label", "Loaded")
        st.markdown(
            f'<div style="text-align: right; padding-top: 6px; font-size: 13px; color: #34D399; font-weight: 600;">'
            f'● Results Loaded ({source_name})'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="text-align: right; padding-top: 6px; font-size: 13px; color: #F59E0B; font-weight: 600;">'
            '● Awaiting CSV Upload'
            '</div>',
            unsafe_allow_html=True
        )

st.markdown('<hr style="border: 0; border-top: 1px solid #334155; margin: 8px 0 20px 0;" />', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Route to Selected View
# ---------------------------------------------------------------------------
if selected_nav == "Experiment Dashboard":
    render_experiment_dashboard()
elif selected_nav == "Model Prediction":
    render_model_prediction()
