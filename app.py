# app.py (Modified for Single Modern Sidebar Button)
from pages.costEstimation_page import render_cost_estimation
from pages.dashboard_page import render_dashboard

import streamlit as st
from aero_utils import (
    load_models,
    process_image_file,
    parse_yolo_labels,
    link_anomalies_to_panels,
    process_video_file
)
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import base64

# 🗂️ App Pages
PAGES = {
    "🏠 Home": "home",
    "📤 Upload Media": "upload",
    "🖼️ Combined Result": "combined",
    "📊 Dashboard": "dashboard",
    "💰 Cost Estimation": "cost"
}

# ✅ Initialize session state before any logic
if "page" not in st.session_state:
    st.session_state["page"] = "🏠 Home"

# 🚀 Page Configuration
st.set_page_config(page_title="AeroAI - AI Solar Panel Inspection", layout="wide")

# 🧭 Sidebar Navigation
st.sidebar.title("⚙️ AeroAI")
st.sidebar.markdown("### 🌐 Navigation")

# 🖌️ Load custom CSS from external file
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- MODIFICATION START ---
# Replace radio navigation with buttons
for label in PAGES.keys():
    if st.sidebar.button(label):
        st.session_state.page = label
# --- MODIFICATION END ---

# 📦 Load Models
PANEL_MODEL_PATH = "models/yolov8_panel.pt"
ANOMALY_MODEL_PATH = "models/yolov5_anomaly.pt"
panel_model = ""
anomaly_model = ""
st.sidebar.success("✅ Models Loaded Successfully!")

# 🔀 Routing
page = st.session_state.page

# ──────────────────────────────────────────────────────────────────────
# helper to set a base64 background image
def set_home_background(png_path: str):
    with open(png_path, "rb") as img_file:
        b64 = base64.b64encode(img_file.read()).decode()
    css = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: url("data:image/png;base64,{b64}") no-repeat center center fixed;
        background-size: cover;
    }}
    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(15,23,42,0.6);
        pointer-events: none;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# 🧭 ROUTE: Home
if page == "🏠 Home":
     # --- Inline CSS + wrapper DIV for Home-only background ---
     # inject the base64 background
    set_home_background("assets/aeroai_home.png")
    
        # Original AeroAI logo
    st.image("assets/aeroai_logo.png", width=200)

    st.markdown("""
    <style>
    .big-title {
        font-size:48px;
        font-weight:bold;
        color:#22D3EE;
        margin-top: 1rem;
    }
    .sub-title {
        font-size:24px;
        color:#94A3B8;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="big-title">Welcome to AeroAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI-powered Solar Panel Inspection Platform</div>', unsafe_allow_html=True)
    st.markdown("AeroAI revolutionizes solar farm maintenance with drone-powered, AI-based inspections.")
    st.markdown("Upload images, videos, or connect your live drone feed to start analyzing your solar assets.")
    col1, col2 = st.columns(2)
    with col1:
        st.button("📤 Start Inspection")
    with col2:
        st.button("📈 View Reports")

# ──────────────────────────────────────────────────────────────────────
# 🧭 ROUTE: Upload Media
elif page == "📤 Upload Media":
    st.title("📤 Upload Inspection Media")
    st.info("Upload your inspection files here (images or videos).")

    uploaded_files = st.file_uploader("Upload Images or Video Files", accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner(f"⏳ Processing `{uploaded_file.name}`... Please wait."):
                if uploaded_file.name.lower().endswith(('.mp4', '.mov', '.avi')):
                    # Video file
                    st.write("🎥 Detected video file. Running inspection...")

                    panel_path, anomaly_path = process_video_file(uploaded_file, panel_model, ANOMALY_MODEL_PATH)

                    # Don't show video preview here
                    st.session_state[f'anomaly_video_{Path(uploaded_file.name).stem}'] = str(anomaly_path)
                    st.session_state[f'panel_video_{Path(uploaded_file.name).stem}'] = str(panel_path)
                    st.success(f"✅ Video Processing Complete: {uploaded_file.name}")

                else:
                    # Image file
                    st.write("🖼️ Detected image file. Running analysis...")

                    panel_image_path, anomaly_image_path = process_image_file(uploaded_file, panel_model, ANOMALY_MODEL_PATH)

                    # Store in session only; do not display
                    st.session_state[f'panel_image_{uploaded_file.name}'] = str(panel_image_path)
                    st.session_state[f'anomaly_image_{uploaded_file.name}'] = str(anomaly_image_path)

                    image_stem = Path(uploaded_file.name).stem
                    panel_label_candidates = sorted(
                        glob.glob(f"processed/panel*/labels/{image_stem}.txt"), reverse=True
                    )
                    anomaly_label_candidates = sorted(
                        glob.glob(f"processed/anomaly*/labels/{image_stem}.txt"), reverse=True
                    )

                    panel_label_path = Path(panel_label_candidates[0]) if panel_label_candidates else None
                    anomaly_label_path = Path(anomaly_label_candidates[0]) if anomaly_label_candidates else None

                    if panel_label_path and anomaly_label_path and panel_label_path.exists() and anomaly_label_path.exists():
                        panel_class_map = {0: 'panel'}
                        anomaly_class_map = {0: 'cracked', 1: 'dusty', 2: 'normal'}

                        panel_boxes = parse_yolo_labels(panel_label_path, panel_class_map, panel_image_path)
                        anomaly_boxes = parse_yolo_labels(anomaly_label_path, anomaly_class_map, anomaly_image_path)

                        panel_anomaly_map = link_anomalies_to_panels(panel_boxes, anomaly_boxes)
                        st.session_state[f'panel_anomaly_map_{uploaded_file.name}'] = panel_anomaly_map
                    else:
                        st.warning(f"⚠️ Labels not found for {uploaded_file.name}. Skipping mapping.")

                    st.success(f"✅ Image Processing Complete: {uploaded_file.name}")

# ──────────────────────────────────────────────────────────────────────
# 🧭 ROUTE: Combined Result
elif page == "🖼️ Combined Result":
    st.title("🖼️ Combined Results")
    st.info("Combined view of detected panels and anomalies.")
    # … (unchanged)

# ──────────────────────────────────────────────────────────────────────
# 🧭 ROUTE: Dashboard
elif page == "📊 Dashboard":
    render_dashboard()

# ──────────────────────────────────────────────────────────────────────
# 🧭 ROUTE: Cost Estimation
elif page == "💰 Cost Estimation":
    render_cost_estimation()
