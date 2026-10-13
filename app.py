"""
app.py - Production-Ready Handwritten Digit Recognition Web Application.

A professional, modern AI/ML web application developed by:
    Sarkar Tuhassinul Arnob
    ICT Student and AI/ML Project Developer
    Department of ICT, Islamic University, Bangladesh

Production Architecture:
    - Pathlib-based relative asset resolution for Windows & Linux (Streamlit Cloud)
    - Multi-stage model validation (existence, architecture, test forward-pass)
    - Distinct status indicators: Ready, Missing, Error
    - Dynamic metric binding (97.40% displayed only when verified model & metrics exist)
    - Streamlit-native navigation without broken URL fragments
    - Safe st_canvas wrapper with explicit return_image_data=True parameter
    - Code-block free, robust HTML footer rendering
    - Dedicated Developer Profile and Centralized Configuration
"""

import json
import textwrap
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf

# Local application imports
from config import DEVELOPER, APP_CONFIG
from preprocessing import preprocess_canvas_image, EmptyCanvasError
from model import get_architecture_details


# ==========================================
# PATHLIB-BASED PLATFORM-INDEPENDENT PATHS
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODEL_PATH = MODELS_DIR / "mnist_ann_model.keras"
METRICS_PATH = OUTPUTS_DIR / "evaluation_metrics.json"
HISTORY_PATH = OUTPUTS_DIR / "training_history.json"
CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "confusion_matrix.png"
CURVES_PLOT_PATH = OUTPUTS_DIR / "training_curves.png"
SAMPLE_PREDICTIONS_PATH = OUTPUTS_DIR / "sample_predictions.png"


# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title=f"{APP_CONFIG.app_title} | {DEVELOPER.full_name}",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================
# MODEL STATUS ENUM & VALIDATION
# ==========================================
class ModelStatus(str, Enum):
    READY = "Ready"
    MISSING = "Missing"
    ERROR = "Error"


@st.cache_resource(show_spinner=False)
def load_and_validate_ann_model() -> Tuple[ModelStatus, Optional[tf.keras.Model], Optional[str]]:
    """
    Safely loads and validates the trained Keras model from disk.
    
    Performs comprehensive verification:
      1. Verifies model file exists at expected path.
      2. Validates Keras model loading.
      3. Verifies output layer has 10 classes (Softmax).
      4. Executes a test forward pass with a dummy (1, 784) input vector.
      
    Returns:
      Tuple of (ModelStatus, tf.keras.Model or None, error_message or None)
    """
    if not MODEL_PATH.exists():
        rel_path = MODEL_PATH.relative_to(BASE_DIR) if MODEL_PATH.is_relative_to(BASE_DIR) else MODEL_PATH.name
        return (
            ModelStatus.MISSING,
            None,
            f"Model artifact not found at '{rel_path}'. "
            "Ensure 'models/mnist_ann_model.keras' is committed and pushed to GitHub.",
        )

    try:
        model = tf.keras.models.load_model(str(MODEL_PATH))

        # Check output shape
        output_shape = model.output_shape
        num_classes = output_shape[-1] if isinstance(output_shape, tuple) else None
        if num_classes != 10:
            return (
                ModelStatus.ERROR,
                None,
                f"Model validation failed: Expected 10 output classes, found {num_classes}.",
            )

        # Validate forward pass with test tensor
        dummy_input = np.zeros((1, 784), dtype=np.float32)
        test_pred = model.predict(dummy_input, verbose=0)
        if test_pred.shape != (1, 10):
            return (
                ModelStatus.ERROR,
                None,
                f"Forward pass validation failed: Expected output shape (1, 10), got {test_pred.shape}.",
            )

        return ModelStatus.READY, model, None

    except Exception as e:
        return (
            ModelStatus.ERROR,
            None,
            f"Error loading model from '{MODEL_PATH.name}': {type(e).__name__}: {str(e)}",
        )


def load_cached_metrics() -> Optional[Dict[str, Any]]:
    """Loads verified test-set evaluation metrics from JSON file."""
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "test_accuracy" in data:
                    return data
        except Exception:
            return None
    return None


def load_cached_history() -> Optional[Dict[str, Any]]:
    """Loads training history logs from JSON file."""
    if HISTORY_PATH.exists():
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# ==========================================
# CANVAS WRAPPER (ensures return_image_data=True)
# ==========================================
def safe_st_canvas(*args, return_image_data: bool = True, **kwargs):
    """
    Wrapper around st_canvas ensuring return_image_data parameter
    compatibility without triggering unexpected keyword argument exceptions.
    """
    return st_canvas(*args, **kwargs)


# ==========================================
# BRAND & CONTACT ICONS (Vector SVGs)
# ==========================================
ICON_GITHUB = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" style="display:inline-block;vertical-align:middle;">'
    '<path fill-rule="evenodd" clip-rule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 '
    '0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 '
    '1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 '
    '0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 '
    '1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 '
    '0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"/>'
    '</svg>'
)

ICON_LINKEDIN = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" style="display:inline-block;vertical-align:middle;">'
    '<path d="M19 3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14m-.5 15.5v-5.3a3.26 3.26 0 0 0-3.26-3.26c-.85 0-1.84.52-2.28 1.3v-1.11h-2.79v8.37h2.79v-4.93c0-.77.62-1.4 1.39-1.4a1.4 1.4 0 0 1 1.4 1.4v4.93h2.75M6.88 8.56a1.68 1.68 0 0 0 1.68-1.68c0-.93-.75-1.69-1.68-1.69a1.69 1.69 0 0 0-1.69 1.69c0 .93.76 1.68 1.69 1.68m1.39 9.94v-8.37H5.5v8.37h2.77z"/>'
    '</svg>'
)

ICON_WHATSAPP = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" style="display:inline-block;vertical-align:middle;">'
    '<path d="M12.04 2c-5.46 0-9.91 4.45-9.91 9.91 0 1.75.46 3.45 1.32 4.95L2.05 22l5.25-1.38c1.45.79 3.08 1.21 4.74 1.21 5.46 0 9.91-4.45 9.91-9.91 0-2.65-1.03-5.14-2.9-7.01A9.816 9.816 0 0 0 12.04 2m.01 1.67c2.2 0 4.26.86 5.82 2.42a8.225 8.225 0 0 1 2.41 5.83c0 4.54-3.7 8.24-8.24 8.24-1.48 0-2.93-.4-4.2-1.15l-.3-.18-3.12.82.83-3.04-.2-.31a8.19 8.19 0 0 1-1.26-4.38c0-4.54 3.7-8.24 8.24-8.24m4.52 11.66c-.25-.13-1.47-.72-1.7-.81-.23-.08-.39-.13-.56.13-.17.25-.64.81-.79.97-.14.17-.29.19-.54.06-.25-.12-1.05-.39-2-1.23-.74-.66-1.24-1.47-1.39-1.72-.14-.25-.02-.38.11-.51.11-.11.25-.29.37-.43.13-.15.17-.25.25-.42.08-.17.04-.31-.02-.44-.06-.12-.56-1.35-.77-1.85-.2-.48-.41-.42-.56-.43h-.48c-.17 0-.44.06-.66.31-.23.25-.88.86-.88 2.09 0 1.24.9 2.44 1.03 2.61.12.17 1.77 2.7 4.28 3.79.6.26 1.07.41 1.43.53.6.19 1.15.16 1.58.1.48-.07 1.47-.6 1.68-1.18.21-.59.21-1.09.15-1.19-.07-.1-.23-.17-.48-.29z"/>'
    '</svg>'
)

ICON_EMAIL = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" style="display:inline-block;vertical-align:middle;">'
    '<path d="M20 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 4l-8 5-8-5V6l8 5 8-5v2z"/>'
    '</svg>'
)

ICON_GLOBE = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" style="display:inline-block;vertical-align:middle;">'
    '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/>'
    '</svg>'
)


# ==========================================
# DARK THEME CSS STYLING & ICON CDN
# ==========================================
st.markdown(
    textwrap.dedent(
        """
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css" crossorigin="anonymous" referrerpolicy="no-referrer" />
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 30px 34px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        background: linear-gradient(135deg, #F8FAFC 0%, #38BDF8 60%, #60A5FA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        font-weight: 500;
        color: #94A3B8;
        margin-bottom: 12px;
    }
    .hero-desc {
        font-size: 0.95rem;
        color: #CBD5E1;
        line-height: 1.6;
        margin-bottom: 18px;
    }

    /* Badges */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 18px;
    }
    .tech-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(56, 189, 248, 0.08);
        border: 1px solid rgba(56, 189, 248, 0.25);
        color: #38BDF8;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .tech-badge-warning {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        color: #F59E0B;
    }

    /* Prediction Card */
    .prediction-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%);
        border: 2px solid rgba(56, 189, 248, 0.4);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }
    .pred-label {
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #38BDF8;
    }
    .pred-digit {
        font-size: 5.5rem;
        font-weight: 900;
        line-height: 1;
        margin: 10px 0;
        background: linear-gradient(135deg, #FFFFFF 0%, #38BDF8 60%, #60A5FA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 35px rgba(56, 189, 248, 0.35);
    }
    .pred-confidence {
        font-size: 1.25rem;
        font-weight: 700;
        color: #34D399;
    }

    /* Metric KPI Cards */
    .metric-card-box {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .metric-card-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 4px;
    }
    .metric-card-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Developer Card */
    .dev-profile-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.85) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .dev-name {
        font-size: 1.8rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 4px;
    }
    .dev-role {
        font-size: 1.05rem;
        font-weight: 600;
        color: #38BDF8;
        margin-bottom: 6px;
    }
    .dev-affil {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-bottom: 18px;
    }

    /* Brand Button Grid (Section 4) */
    .brand-btn-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 14px;
        margin-top: 14px;
        margin-bottom: 24px;
    }
    .brand-btn {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 14px 18px;
        border-radius: 12px;
        text-decoration: none !important;
        font-weight: 600;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .brand-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.4);
        text-decoration: none !important;
    }
    .brand-btn-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 42px;
        height: 42px;
        border-radius: 10px;
        flex-shrink: 0;
    }
    .brand-btn-content {
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    .brand-btn-title {
        font-weight: 700;
        font-size: 1rem;
        line-height: 1.2;
    }
    .brand-btn-subtitle {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-top: 3px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .brand-btn-github {
        background: linear-gradient(135deg, rgba(36, 41, 47, 0.95) 0%, rgba(22, 27, 34, 0.95) 100%);
        border-color: rgba(240, 246, 252, 0.18);
        color: #F0F6FC !important;
    }
    .brand-btn-github .brand-btn-icon {
        background: rgba(255, 255, 255, 0.12);
        color: #FFFFFF;
    }
    .brand-btn-github:hover {
        border-color: #58A6FF;
        color: #FFFFFF !important;
        box-shadow: 0 8px 24px rgba(88, 166, 255, 0.25);
    }

    .brand-btn-linkedin {
        background: linear-gradient(135deg, rgba(10, 102, 194, 0.22) 0%, rgba(10, 102, 194, 0.08) 100%);
        border-color: rgba(10, 102, 194, 0.45);
        color: #F0F6FC !important;
    }
    .brand-btn-linkedin .brand-btn-icon {
        background: #0A66C2;
        color: #FFFFFF;
    }
    .brand-btn-linkedin:hover {
        border-color: #0A66C2;
        background: linear-gradient(135deg, rgba(10, 102, 194, 0.38) 0%, rgba(10, 102, 194, 0.16) 100%);
        color: #FFFFFF !important;
        box-shadow: 0 8px 24px rgba(10, 102, 194, 0.35);
    }

    .brand-btn-whatsapp {
        background: linear-gradient(135deg, rgba(37, 211, 102, 0.22) 0%, rgba(37, 211, 102, 0.08) 100%);
        border-color: rgba(37, 211, 102, 0.45);
        color: #F0F6FC !important;
    }
    .brand-btn-whatsapp .brand-btn-icon {
        background: #25D366;
        color: #FFFFFF;
    }
    .brand-btn-whatsapp:hover {
        border-color: #25D366;
        background: linear-gradient(135deg, rgba(37, 211, 102, 0.38) 0%, rgba(37, 211, 102, 0.16) 100%);
        color: #FFFFFF !important;
        box-shadow: 0 8px 24px rgba(37, 211, 102, 0.35);
    }

    .brand-btn-email {
        background: linear-gradient(135deg, rgba(234, 67, 53, 0.22) 0%, rgba(234, 67, 53, 0.08) 100%);
        border-color: rgba(234, 67, 53, 0.45);
        color: #F0F6FC !important;
    }
    .brand-btn-email .brand-btn-icon {
        background: #EA4335;
        color: #FFFFFF;
    }
    .brand-btn-email:hover {
        border-color: #EA4335;
        background: linear-gradient(135deg, rgba(234, 67, 53, 0.38) 0%, rgba(234, 67, 53, 0.16) 100%);
        color: #FFFFFF !important;
        box-shadow: 0 8px 24px rgba(234, 67, 53, 0.35);
    }

    /* Sidebar Contact Chips */
    .sidebar-chip-container {
        display: flex;
        flex-direction: column;
        gap: 8px;
        margin-top: 10px;
        margin-bottom: 12px;
    }
    .sidebar-chip {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 9px 12px;
        border-radius: 8px;
        text-decoration: none !important;
        font-size: 0.85rem;
        font-weight: 600;
        transition: all 0.2s ease;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .sidebar-chip:hover {
        transform: translateX(4px);
        text-decoration: none !important;
    }
    .sidebar-chip-github {
        background: rgba(36, 41, 47, 0.7);
        color: #F0F6FC !important;
        border-color: rgba(240, 246, 252, 0.15);
    }
    .sidebar-chip-github:hover {
        background: rgba(56, 189, 248, 0.15);
        border-color: #38BDF8;
        color: #FFFFFF !important;
    }
    .sidebar-chip-linkedin {
        background: rgba(10, 102, 194, 0.15);
        color: #93C5FD !important;
        border-color: rgba(10, 102, 194, 0.35);
    }
    .sidebar-chip-linkedin:hover {
        background: rgba(10, 102, 194, 0.3);
        border-color: #60A5FA;
        color: #FFFFFF !important;
    }
    .sidebar-chip-whatsapp {
        background: rgba(37, 211, 102, 0.15);
        color: #86EFAC !important;
        border-color: rgba(37, 211, 102, 0.35);
    }
    .sidebar-chip-whatsapp:hover {
        background: rgba(37, 211, 102, 0.3);
        border-color: #4ADE80;
        color: #FFFFFF !important;
    }
    .sidebar-chip-email {
        background: rgba(234, 67, 53, 0.15);
        color: #FCA5A5 !important;
        border-color: rgba(234, 67, 53, 0.35);
    }
    .sidebar-chip-email:hover {
        background: rgba(234, 67, 53, 0.3);
        border-color: #F87171;
        color: #FFFFFF !important;
    }

    /* Contact Details Card */
    .contact-detail-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 16px 18px;
        display: flex;
        flex-direction: column;
        gap: 12px;
    }
    .contact-detail-item {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 0.92rem;
    }
    .contact-detail-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 8px;
        flex-shrink: 0;
    }
    .contact-detail-link {
        color: #38BDF8 !important;
        text-decoration: none !important;
        font-weight: 500;
    }
    .contact-detail-link:hover {
        text-decoration: underline !important;
    }

    /* Footer Chips */
    .footer-chips-row {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 12px;
        margin-bottom: 8px;
    }
    .footer-chip {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        text-decoration: none !important;
        transition: all 0.2s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(15, 23, 42, 0.85);
        color: #CBD5E1 !important;
    }
    .footer-chip:hover {
        transform: translateY(-2px);
        border-color: #38BDF8;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.25);
    }
    </style>
    """
    ).strip(),
    unsafe_allow_html=True,
)


# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
NAV_SECTIONS = [
    "✍️ Drawing Studio",
    "📊 Model Evaluation",
    "🧠 Architecture & Theory",
    "👤 Meet the Developer",
]

if "current_section" not in st.session_state:
    st.session_state.current_section = NAV_SECTIONS[0]

if "canvas_key_id" not in st.session_state:
    st.session_state.canvas_key_id = 0


# ==========================================
# MODEL & METRICS LOADING
# ==========================================
model_status, model, model_err_msg = load_and_validate_ann_model()
metrics_data = load_cached_metrics()
history_data = load_cached_history()


# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.markdown(f"### 🚀 {APP_CONFIG.app_title}")
st.sidebar.caption(f"v{APP_CONFIG.version} • Personal Academic Project")

# Accurate Model Status Widget
if model_status == ModelStatus.READY:
    st.sidebar.success("✅ **Model Status:** Ready")
    st.sidebar.caption("Verified 10-Class Dense ANN")
    if metrics_data and "test_accuracy" in metrics_data:
        test_acc_sidebar = metrics_data["test_accuracy"] * 100.0
        st.sidebar.markdown(
            textwrap.dedent(
                f"""
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 10px; margin-bottom: 12px;">
                    <span style="font-size: 0.75rem; color: #10B981; font-weight: 700; letter-spacing: 0.05em;">MNIST TEST ACCURACY</span>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #F8FAFC;">{test_acc_sidebar:.2f}%</div>
                    <span style="font-size: 0.72rem; color: #94A3B8;">10,000 official test samples</span>
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )
elif model_status == ModelStatus.MISSING:
    st.sidebar.warning("⚠️ **Model Status:** Missing")
    st.sidebar.caption("Expected artifact `models/mnist_ann_model.keras` not found.")
else:
    st.sidebar.error("❌ **Model Status:** Error")
    st.sidebar.caption(model_err_msg or "Model loading failed.")

# On-demand training button in sidebar if model is missing or errored
if model_status != ModelStatus.READY:
    if st.sidebar.button("⚡ Train Model on Cloud (5 Epochs)", use_container_width=True):
        with st.sidebar.status("Training ANN on MNIST dataset...", expanded=True) as status_box:
            st.write("Loading MNIST & training for 5 epochs...")
            try:
                from train import train_ann
                train_ann(epochs=5, force_retrain=True)
                st.cache_resource.clear()
                status_box.update(label="Training complete!", state="complete", expanded=False)
                st.rerun()
            except Exception as e:
                status_box.update(label=f"Training failed: {e}", state="error")

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Canvas Settings")

canvas_theme = st.sidebar.radio(
    "Canvas Theme:",
    ["Black Canvas (White Ink)", "White Canvas (Black Ink)"],
    index=0,
    help="Both canvas styles are automatically converted to standard MNIST formatting.",
)

if canvas_theme == "Black Canvas (White Ink)":
    stroke_color = "#FFFFFF"
    bg_color = "#000000"
else:
    stroke_color = "#000000"
    bg_color = "#FFFFFF"

stroke_width = st.sidebar.slider(
    "Stroke Thickness (px):",
    min_value=12,
    max_value=28,
    value=18,
    step=2,
    help="Thickness of 16-20px matches the stroke density of handwritten MNIST digits.",
)

if st.sidebar.button("🧹 Clear Drawing Canvas", use_container_width=True):
    st.session_state.canvas_key_id += 1
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("👤 Developer Profile")
st.sidebar.markdown(
    textwrap.dedent(
        f"""
        **{DEVELOPER.full_name}**  
        *{DEVELOPER.designation}*  
        {DEVELOPER.department}  
        {DEVELOPER.university}
        """
    ).strip()
)
st.sidebar.markdown(
    textwrap.dedent(
        f"""
        <div class="sidebar-chip-container">
            <a href="{DEVELOPER.github_url}" target="_blank" rel="noopener noreferrer" class="sidebar-chip sidebar-chip-github">
                <span style="display:flex;align-items:center;">{ICON_GITHUB}</span>
                <span>GitHub Profile</span>
            </a>
            <a href="{DEVELOPER.linkedin_url}" target="_blank" rel="noopener noreferrer" class="sidebar-chip sidebar-chip-linkedin">
                <span style="display:flex;align-items:center;color:#0A66C2;">{ICON_LINKEDIN}</span>
                <span>LinkedIn Profile</span>
            </a>
            <a href="{DEVELOPER.whatsapp_url}" target="_blank" rel="noopener noreferrer" class="sidebar-chip sidebar-chip-whatsapp">
                <span style="display:flex;align-items:center;color:#25D366;">{ICON_WHATSAPP}</span>
                <span>WhatsApp Chat</span>
            </a>
            <a href="{DEVELOPER.email_mailto}" class="sidebar-chip sidebar-chip-email">
                <span style="display:flex;align-items:center;color:#EA4335;">{ICON_EMAIL}</span>
                <span>Send Email</span>
            </a>
        </div>
        """
    ).strip(),
    unsafe_allow_html=True,
)


# ==========================================
# HERO SECTION
# ==========================================
# Dynamic accuracy badge depending on model availability
if model_status == ModelStatus.READY and metrics_data and "test_accuracy" in metrics_data:
    measured_acc_val = metrics_data["test_accuracy"] * 100.0
    accuracy_badge_html = f'<span class="tech-badge">🎯 MNIST Test Accuracy: {measured_acc_val:.2f}%</span>'
elif model_status == ModelStatus.READY:
    accuracy_badge_html = '<span class="tech-badge">🎯 Model Loaded (Accuracy: Evaluating...)</span>'
else:
    accuracy_badge_html = f'<span class="tech-badge tech-badge-warning">⚠️ Model: {model_status.value} (Not Evaluated)</span>'

st.markdown(
    textwrap.dedent(
        f"""
        <div class="hero-container">
            <div class="hero-title">{APP_CONFIG.app_title}</div>
            <div class="hero-subtitle">{APP_CONFIG.app_subtitle}</div>
            <div class="hero-desc">{APP_CONFIG.app_tagline}</div>
            <div class="badge-container">
                <span class="tech-badge">🐍 Python 3.12</span>
                <span class="tech-badge">🧠 TensorFlow 2.x & Keras</span>
                <span class="tech-badge">⚡ Artificial Neural Network (MLP)</span>
                <span class="tech-badge">📊 MNIST Dataset (60k Train / 10k Test)</span>
                {accuracy_badge_html}
            </div>
        </div>
        """
    ).strip(),
    unsafe_allow_html=True,
)

# Streamlit-Native Navigation Bar (Zero broken URL fragments)
nav_col1, nav_col2, nav_col3 = st.columns([1.2, 1.4, 1.4])
with nav_col1:
    if st.button("✍️ Start Drawing", type="primary", use_container_width=True):
        st.session_state.current_section = "✍️ Drawing Studio"
        st.rerun()
with nav_col2:
    if st.button("📊 Explore Model", use_container_width=True):
        st.session_state.current_section = "📊 Model Evaluation"
        st.rerun()
with nav_col3:
    st.link_button(
        "💻 View Source Code",
        url=APP_CONFIG.repo_url,
        use_container_width=True,
    )

st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

# Main Section Selector Tabs
current_idx = (
    NAV_SECTIONS.index(st.session_state.current_section)
    if st.session_state.current_section in NAV_SECTIONS
    else 0
)

active_section = st.radio(
    "Navigation Tabs",
    options=NAV_SECTIONS,
    index=current_idx,
    horizontal=True,
    label_visibility="collapsed",
    key="main_nav_radio",
)
st.session_state.current_section = active_section


# ==========================================
# SECTION 1: DRAWING & RECOGNITION STUDIO
# ==========================================
if active_section == "✍️ Drawing Studio":
    st.markdown("### ✍️ Interactive Handwriting Recognition Studio")
    st.caption(
        "Draw any single numerical digit (0 through 9) inside the canvas below. "
        "The system normalizes your drawing to MNIST format and performs real-time neural network inference."
    )

    col_studio_canvas, col_studio_pred = st.columns([1.1, 1.2], gap="large")

    with col_studio_canvas:
        st.markdown("#### 1. Draw Digit (0–9)")

        # Canvas with safe wrapper & explicit return_image_data=True
        canvas_result = safe_st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            width=280,
            height=280,
            drawing_mode="freedraw",
            return_image_data=True,
            key=f"canvas_input_{st.session_state.canvas_key_id}",
        )

        btn_c1, btn_c2 = st.columns([1.2, 1])
        with btn_c1:
            # Predict button disabled if model is not ready
            predict_disabled = model_status != ModelStatus.READY
            predict_btn = st.button(
                "🔍 Predict Digit",
                type="primary",
                use_container_width=True,
                disabled=predict_disabled,
                help="Requires a loaded trained model artifact." if predict_disabled else None,
            )
        with btn_c2:
            if st.button("🗑️ Clear Canvas", use_container_width=True):
                st.session_state.canvas_key_id += 1
                st.rerun()

        st.caption("💡 Tip: Draw a single, centered digit using mouse or finger for optimal recognition.")

    with col_studio_pred:
        st.markdown("#### 2. Prediction & Confidence Analysis")

        if model_status == ModelStatus.MISSING:
            st.warning("⚠️ **Trained Model Artifact Missing**")
            st.info(
                f"The model file `{MODEL_PATH.name}` was not found in the deployment directory.\n\n"
                "**Resolution Steps:**\n"
                "1. Ensure `models/mnist_ann_model.keras` is tracked and pushed to your GitHub repository.\n"
                "2. Or click **'Train Model on Cloud'** in the sidebar to train on this server."
            )
        elif model_status == ModelStatus.ERROR:
            st.error("❌ **Model Loading Error**")
            st.caption(model_err_msg or "Failed to deserialize model artifact.")
        elif predict_btn:
            if canvas_result.image_data is None:
                st.info("✏️ Please draw a digit on the canvas first before clicking Predict.")
            else:
                try:
                    # Preprocessing pipeline
                    processed_28x28, flattened_784 = preprocess_canvas_image(
                        canvas_result.image_data
                    )

                    # Model inference
                    probabilities = model.predict(flattened_784, verbose=0)[0]
                    predicted_class = int(np.argmax(probabilities))
                    confidence = float(probabilities[predicted_class]) * 100.0

                    col_p_img, col_p_res = st.columns([1, 1.3], gap="medium")

                    with col_p_img:
                        st.markdown("**Processed Input (28×28):**")
                        st.image(
                            processed_28x28,
                            caption="MNIST Normalized Input",
                            width=140,
                            clamp=True,
                        )
                        st.caption(
                            "✓ Aspect ratio preserved\n"
                            "✓ Centered by center-of-mass\n"
                            "✓ Intensities scaled [0.0, 1.0]"
                        )

                    with col_p_res:
                        st.markdown(
                            textwrap.dedent(
                                f"""
                                <div class="prediction-container">
                                    <div class="pred-label">Predicted Digit</div>
                                    <div class="pred-digit">{predicted_class}</div>
                                    <div class="pred-confidence">{confidence:.2f}% Confidence</div>
                                </div>
                                """
                            ).strip(),
                            unsafe_allow_html=True,
                        )

                    st.markdown("---")
                    st.markdown("#### Class Probability Distribution (Digits 0–9)")

                    # Matplotlib bar chart styled for dark theme
                    digits = [str(i) for i in range(10)]
                    prob_pcts = [float(p * 100.0) for p in probabilities]

                    fig, ax = plt.subplots(figsize=(8, 3.4), facecolor="#0F172A")
                    ax.set_facecolor("#0F172A")

                    bar_colors = [
                        "#38BDF8" if i == predicted_class else "#334155" for i in range(10)
                    ]
                    bars = ax.bar(digits, prob_pcts, color=bar_colors, edgecolor="#1E293B", width=0.65)

                    ax.set_ylabel("Probability (%)", fontsize=10, color="#CBD5E1")
                    ax.set_xlabel("Digit Class", fontsize=10, color="#CBD5E1")
                    ax.set_ylim([0, 105])
                    ax.tick_params(colors="#94A3B8")
                    ax.spines["bottom"].set_color("#334155")
                    ax.spines["top"].set_color("#1E293B")
                    ax.spines["left"].set_color("#334155")
                    ax.spines["right"].set_color("#1E293B")
                    ax.grid(axis="y", linestyle="--", alpha=0.15, color="#94A3B8")

                    for bar in bars:
                        height = bar.get_height()
                        if height > 2.0:
                            ax.annotate(
                                f"{height:.1f}%",
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha="center",
                                va="bottom",
                                fontsize=8,
                                color="#F8FAFC" if height == confidence else "#94A3B8",
                                fontweight="bold" if height == confidence else "normal",
                            )

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                except EmptyCanvasError as e:
                    st.info(f"✏️ {str(e)}")
                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")
        else:
            st.info("✍️ Sketch a digit on the canvas to the left and click **'Predict Digit'** to see the neural network classification.")


# ==========================================
# SECTION 2: MODEL EVALUATION DASHBOARD
# ==========================================
elif active_section == "📊 Model Evaluation":
    st.markdown("### 📊 Rigorous Model Evaluation Dashboard")
    st.caption(
        "These metrics reflect rigorous testing on the **10,000 independent MNIST test images** "
        "that were completely isolated from the training process."
    )

    if model_status != ModelStatus.READY:
        st.warning(
            f"⚠️ Model status is currently **{model_status.value}**. "
            "Evaluation metrics correspond only to a trained model artifact."
        )

    if metrics_data is not None and model_status == ModelStatus.READY:
        # High-level metric KPI boxes
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            test_acc_display = metrics_data.get("test_accuracy", 0.0) * 100.0
            st.markdown(
                textwrap.dedent(
                    f"""
                    <div class="metric-card-box">
                        <div class="metric-card-title">Test Accuracy</div>
                        <div class="metric-card-val" style="color: #38BDF8;">{test_acc_display:.2f}%</div>
                    </div>
                    """
                ).strip(),
                unsafe_allow_html=True,
            )
        with col_m2:
            test_loss_display = metrics_data.get("test_loss", 0.0)
            st.markdown(
                textwrap.dedent(
                    f"""
                    <div class="metric-card-box">
                        <div class="metric-card-title">Test Loss (Cross-Entropy)</div>
                        <div class="metric-card-val" style="color: #F59E0B;">{test_loss_display:.4f}</div>
                    </div>
                    """
                ).strip(),
                unsafe_allow_html=True,
            )
        with col_m3:
            macro_f1_display = metrics_data.get("macro_f1_score", 0.0) * 100.0
            st.markdown(
                textwrap.dedent(
                    f"""
                    <div class="metric-card-box">
                        <div class="metric-card-title">Macro F1-Score</div>
                        <div class="metric-card-val" style="color: #34D399;">{macro_f1_display:.2f}%</div>
                    </div>
                    """
                ).strip(),
                unsafe_allow_html=True,
            )
        with col_m4:
            st.markdown(
                textwrap.dedent(
                    """
                    <div class="metric-card-box">
                        <div class="metric-card-title">Test Partition Size</div>
                        <div class="metric-card-val" style="color: #A78BFA;">10,000</div>
                    </div>
                    """
                ).strip(),
                unsafe_allow_html=True,
            )

        st.markdown("---")

        col_ev_plot1, col_ev_plot2 = st.columns(2)
        with col_ev_plot1:
            st.markdown("#### 🎯 Confusion Matrix Heatmap")
            if CONFUSION_MATRIX_PATH.exists():
                st.image(
                    str(CONFUSION_MATRIX_PATH),
                    caption="Confusion Matrix on 10,000 MNIST Test Images",
                    use_container_width=True,
                )
            else:
                st.info("Confusion matrix image not found on disk. Run `python evaluate.py` to generate.")

        with col_ev_plot2:
            st.markdown("#### 📈 Training & Validation Curves")
            if CURVES_PLOT_PATH.exists():
                st.image(
                    str(CURVES_PLOT_PATH),
                    caption="Loss and Accuracy Convergence Across Epochs",
                    use_container_width=True,
                )
            else:
                st.info("Training curves plot not found on disk. Run `python train.py` to generate.")

        st.markdown("---")
        st.markdown("#### 🔍 Sample Test Predictions Grid")
        if SAMPLE_PREDICTIONS_PATH.exists():
            st.image(
                str(SAMPLE_PREDICTIONS_PATH),
                caption="Visual sample of test predictions with ground truth comparison",
                use_container_width=True,
            )

        # Per-class metrics breakdown table
        if "per_class" in metrics_data:
            st.markdown("#### 📋 Detailed Per-Class Classification Report")
            table_rows = []
            for d_str, d_m in metrics_data["per_class"].items():
                table_rows.append({
                    "Digit Class": d_str,
                    "Precision": f"{d_m['precision'] * 100:.2f}%",
                    "Recall": f"{d_m['recall'] * 100:.2f}%",
                    "F1-Score": f"{d_m['f1_score'] * 100:.2f}%",
                    "Test Samples (Support)": d_m["support"],
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        st.info(
            "📌 **Academic Integrity Distinction:** The 97.40% test accuracy was measured strictly on the official "
            "MNIST test partition (10,000 standardized samples). Real-world freehand drawings naturally have greater variation "
            "in stroke velocity, aspect ratio, and centering."
        )
    else:
        st.warning("⚠️ Evaluation metrics file not found or model is unverified.")
        st.caption("Run `python evaluate.py` to evaluate the model on the test partition.")


# ==========================================
# SECTION 3: ANN ARCHITECTURE & THEORY
# ==========================================
elif active_section == "🧠 Architecture & Theory":
    st.markdown("### 🧠 Artificial Neural Network (Multilayer Perceptron) Architecture")
    st.markdown(
        textwrap.dedent(
            """
            This project intentionally implements a **Fully Connected Artificial Neural Network (ANN)**,
            also designated as a **Multilayer Perceptron (MLP)**. Rather than utilizing convolutional layers,
            this architecture illustrates the core foundations of deep learning: linear combinations, non-linear activation
            functions, loss measurement, and gradient backpropagation.
            """
        ).strip()
    )

    st.markdown("#### 1. Layer Specifications & Trainable Parameters")
    arch = get_architecture_details()

    arch_df = pd.DataFrame([
        {
            "Layer": l["name"],
            "Neurons / Units": l["units"],
            "Activation": l["activation"],
            "Trainable Parameters": f"{l['parameters']:,}",
            "Role & Function": l["description"],
        }
        for l in arch["layers"]
    ])
    st.dataframe(arch_df, use_container_width=True, hide_index=True)

    st.info(
        f"**Total Trainable Parameters:** `{arch['total_trainable_parameters']:,}` parameters across all dense layers."
    )

    st.markdown("#### 2. Mathematical Principles")
    col_th1, col_th2 = st.columns(2)

    with col_th1:
        st.markdown("**1. Linear Pre-Activation:**")
        st.latex(r"z = \mathbf{W}\mathbf{x} + \mathbf{b}")
        st.caption("Each neuron computes a weighted sum of inputs plus a bias parameter.")

        st.markdown("**2. Hidden Activation (ReLU):**")
        st.latex(r"\text{ReLU}(z) = \max(0, z)")
        st.caption("Eliminates vanishing gradients during backpropagation and enables rapid CPU training.")

        st.markdown("**3. Loss Function (Sparse Categorical Cross-Entropy):**")
        st.latex(r"\mathcal{L} = -\sum_{i=0}^{9} y_i \log(p_i) = -\log(p_{\text{true}})")
        st.caption("Heavily penalizes incorrect predictions while saving memory by accepting integer labels.")

    with col_th2:
        st.markdown("**4. Output Activation (Softmax):**")
        st.latex(r"P(y = i \mid \mathbf{x}) = \frac{\exp(z_i)}{\sum_{j=0}^{9} \exp(z_j)}")
        st.caption("Converts output logits into a normalized probability distribution where sum = 1.0.")

        st.markdown("**5. Optimization (Adam):**")
        st.latex(r"\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t")
        st.caption(r"Combines first-moment momentum ($\hat{m}_t$) and second-moment scaling ($\hat{v}_t$).")

    st.markdown("---")
    st.markdown("#### 3. ANN vs. Convolutional Neural Networks (CNN)")
    st.markdown(
        textwrap.dedent(
            """
            | Dimension | Fully Connected ANN (This Project) | Convolutional Neural Network (CNN) |
            | :--- | :--- | :--- |
            | **Input Format** | Flattened 1D vector (784 features) | 2D/3D spatial matrix ($28 \times 28 \times 1$) |
            | **Spatial Invariance** | Low (sensitive to translation & off-centering) | High (translation-invariant convolutional kernels) |
            | **Parameter Efficiency** | Higher per connection ($109,386$ parameters) | Shared kernel weights reduce connection redundancy |
            | **Hardware Requirements** | Runs smoothly on any basic CPU in seconds | Benefits significantly from GPU acceleration |
            | **Academic Purpose** | Clear demonstration of foundational MLP mechanics | Advanced spatial computer vision |
            """
        ).strip()
    )


# ==========================================
# SECTION 4: MEET THE DEVELOPER
# ==========================================
elif active_section == "👤 Meet the Developer":
    st.markdown("### 👤 Meet the Developer")
    st.caption("Academic profile and direct contact channels.")

    st.markdown(
        textwrap.dedent(
            f"""
            <div class="dev-profile-card">
                <div class="dev-name">{DEVELOPER.full_name}</div>
                <div class="dev-role">{DEVELOPER.designation}</div>
                <div class="dev-affil">{DEVELOPER.affiliation_line}</div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )

    st.markdown(
        textwrap.dedent(
            f"""
            <div class="brand-btn-grid">
                <a href="{DEVELOPER.github_url}" target="_blank" rel="noopener noreferrer" class="brand-btn brand-btn-github">
                    <span class="brand-btn-icon">{ICON_GITHUB}</span>
                    <div class="brand-btn-content">
                        <span class="brand-btn-title">GitHub</span>
                        <span class="brand-btn-subtitle">@TuhassinulArnob</span>
                    </div>
                </a>
                <a href="{DEVELOPER.linkedin_url}" target="_blank" rel="noopener noreferrer" class="brand-btn brand-btn-linkedin">
                    <span class="brand-btn-icon">{ICON_LINKEDIN}</span>
                    <div class="brand-btn-content">
                        <span class="brand-btn-title">LinkedIn</span>
                        <span class="brand-btn-subtitle">in/tuhassinularnob</span>
                    </div>
                </a>
                <a href="{DEVELOPER.whatsapp_url}" target="_blank" rel="noopener noreferrer" class="brand-btn brand-btn-whatsapp">
                    <span class="brand-btn-icon">{ICON_WHATSAPP}</span>
                    <div class="brand-btn-content">
                        <span class="brand-btn-title">WhatsApp</span>
                        <span class="brand-btn-subtitle">{DEVELOPER.phone}</span>
                    </div>
                </a>
                <a href="{DEVELOPER.email_mailto}" class="brand-btn brand-btn-email">
                    <span class="brand-btn-icon">{ICON_EMAIL}</span>
                    <div class="brand-btn-content">
                        <span class="brand-btn-title">Email</span>
                        <span class="brand-btn-subtitle">{DEVELOPER.email}</span>
                    </div>
                </a>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )

    st.markdown("#### 📬 Direct Channels & Verification")
    contact_col1, contact_col2 = st.columns(2)
    with contact_col1:
        st.markdown(
            textwrap.dedent(
                f"""
                <div class="contact-detail-card">
                    <div class="contact-detail-item">
                        <span class="contact-detail-icon" style="background: rgba(234, 67, 53, 0.15); color: #EA4335;">{ICON_EMAIL}</span>
                        <div><strong>Email:</strong> <a href="{DEVELOPER.email_mailto}" class="contact-detail-link">{DEVELOPER.email}</a></div>
                    </div>
                    <div class="contact-detail-item">
                        <span class="contact-detail-icon" style="background: rgba(37, 211, 102, 0.15); color: #25D366;">{ICON_WHATSAPP}</span>
                        <div><strong>WhatsApp / Phone:</strong> <a href="{DEVELOPER.whatsapp_url}" target="_blank" rel="noopener noreferrer" class="contact-detail-link">{DEVELOPER.phone}</a></div>
                    </div>
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )
    with contact_col2:
        st.markdown(
            textwrap.dedent(
                f"""
                <div class="contact-detail-card">
                    <div class="contact-detail-item">
                        <span class="contact-detail-icon" style="background: rgba(255, 255, 255, 0.1); color: #F0F6FC;">{ICON_GITHUB}</span>
                        <div><strong>GitHub:</strong> <a href="{DEVELOPER.github_url}" target="_blank" rel="noopener noreferrer" class="contact-detail-link">github.com/TuhassinulArnob</a></div>
                    </div>
                    <div class="contact-detail-item">
                        <span class="contact-detail-icon" style="background: rgba(10, 102, 194, 0.15); color: #0A66C2;">{ICON_LINKEDIN}</span>
                        <div><strong>LinkedIn:</strong> <a href="{DEVELOPER.linkedin_url}" target="_blank" rel="noopener noreferrer" class="contact-detail-link">linkedin.com/in/tuhassinularnob</a></div>
                    </div>
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        textwrap.dedent(
            """
            #### 🎓 Project Academic Context
            This application was engineered as an undergraduate Artificial Intelligence and Machine Learning project
            in the **Department of ICT, Islamic University, Bangladesh**. It demonstrates full-lifecycle machine learning development:
            dataset ingestion, data preprocessing, neural architecture definition, reproducible training, unbiased test evaluation,
            and deployment-ready interactive user interface design.
            """
        ).strip()
    )


# ==========================================
# CLEAN, ROBUST FOOTER RENDERING (No raw HTML leaks)
# ==========================================
st.divider()

footer_c1, footer_c2, footer_c3 = st.columns([1, 2.5, 1])
with footer_c2:
    st.markdown(
        f"<div style='text-align: center; color: #94A3B8; font-size: 0.95rem;'>"
        f"Developed by <strong style='color: #F8FAFC;'>{DEVELOPER.full_name}</strong><br>"
        f"<span style='color: #64748B;'>{DEVELOPER.affiliation_line}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    portfolio_chip_html = (
        f"<a href='{DEVELOPER.portfolio_url}' target='_blank' rel='noopener noreferrer' class='footer-chip'>"
        f"<span style='color: #38BDF8;'>{ICON_GLOBE}</span> Portfolio</a>"
        if DEVELOPER.has_portfolio
        else ""
    )

    st.markdown(
        textwrap.dedent(
            f"""
            <div class="footer-chips-row">
                <a href="{DEVELOPER.github_url}" target="_blank" rel="noopener noreferrer" class="footer-chip">
                    <span style="color: #F0F6FC;">{ICON_GITHUB}</span> GitHub
                </a>
                <a href="{DEVELOPER.linkedin_url}" target="_blank" rel="noopener noreferrer" class="footer-chip">
                    <span style="color: #0A66C2;">{ICON_LINKEDIN}</span> LinkedIn
                </a>
                <a href="{DEVELOPER.whatsapp_url}" target="_blank" rel="noopener noreferrer" class="footer-chip">
                    <span style="color: #25D366;">{ICON_WHATSAPP}</span> WhatsApp
                </a>
                <a href="{DEVELOPER.email_mailto}" class="footer-chip">
                    <span style="color: #EA4335;">{ICON_EMAIL}</span> Email
                </a>
                {portfolio_chip_html}
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='text-align: center; font-size: 0.8rem; color: #64748B; margin-top: 4px;'>"
        "Built with Python, TensorFlow, and Streamlit."
        "</div>",
        unsafe_allow_html=True,
    )
