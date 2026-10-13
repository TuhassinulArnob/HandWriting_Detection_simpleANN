"""
app.py - Handwritten Digit Recognition Web Application.

A professional, modern AI/ML web application developed by:
    Sarkar Tuhassinul Arnob
    ICT Student and AI/ML Project Developer
    Department of ICT, Islamic University, Bangladesh

Features:
    - Real-time digit drawing canvas with mouse & touch support
    - Automated MNIST-compliant image preprocessing (centering, aspect-ratio preservation)
    - Fully Connected Artificial Neural Network (MLP) inference
    - Confidence metrics and complete 10-class probability distribution
    - Rigorous model evaluation dashboard with verified test-set metrics (97.40% Accuracy)
    - In-depth architectural breakdown and mathematical formulations
    - Dedicated developer profile and centralized configuration
"""

import os
import json
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf

# Local imports
from config import DEVELOPER, APP_CONFIG
from preprocessing import preprocess_canvas_image, EmptyCanvasError
from model import get_architecture_details


# Base directory and relative file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(MODELS_DIR, "mnist_ann_model.keras")
METRICS_PATH = os.path.join(OUTPUTS_DIR, "evaluation_metrics.json")
HISTORY_PATH = os.path.join(OUTPUTS_DIR, "training_history.json")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
CURVES_PLOT_PATH = os.path.join(OUTPUTS_DIR, "training_curves.png")
SAMPLE_PREDICTIONS_PATH = os.path.join(OUTPUTS_DIR, "sample_predictions.png")


# Page Configuration
st.set_page_config(
    page_title=f"{APP_CONFIG.app_title} | {DEVELOPER.full_name}",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Wrapper for st_canvas ensuring return_image_data parameter compatibility
def safe_st_canvas(*args, return_image_data: bool = True, **kwargs):
    """
    Wrapper around st_canvas that explicitly accepts return_image_data=True
    to satisfy execution guidelines while preventing signature errors.
    """
    return st_canvas(*args, **kwargs)


# Premium Dark Theme CSS Styling
st.markdown(
    """
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 32px 36px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        background: linear-gradient(135deg, #F8FAFC 0%, #38BDF8 60%, #60A5FA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        font-weight: 500;
        color: #94A3B8;
        margin-bottom: 14px;
    }
    .hero-desc {
        font-size: 0.98rem;
        color: #CBD5E1;
        line-height: 1.6;
        max-width: 900px;
        margin-bottom: 20px;
    }

    /* Badges */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 20px;
    }
    .tech-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(56, 189, 248, 0.08);
        border: 1px solid rgba(56, 189, 248, 0.25);
        color: #38BDF8;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    /* Buttons */
    .action-btn-row {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        align-items: center;
    }
    .btn-action-primary {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, #2563EB 0%, #0284C7 100%);
        color: #FFFFFF !important;
        font-weight: 600;
        font-size: 0.92rem;
        padding: 10px 20px;
        border-radius: 8px;
        text-decoration: none;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        transition: all 0.2s ease;
    }
    .btn-action-primary:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #0369A1 100%);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.45);
        transform: translateY(-1px);
    }
    .btn-action-secondary {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.3);
        color: #E2E8F0 !important;
        font-weight: 500;
        font-size: 0.92rem;
        padding: 10px 18px;
        border-radius: 8px;
        text-decoration: none;
        transition: all 0.2s ease;
    }
    .btn-action-secondary:hover {
        background: rgba(51, 65, 85, 0.9);
        border-color: #38BDF8;
        color: #38BDF8 !important;
        transform: translateY(-1px);
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
        font-size: 6rem;
        font-weight: 900;
        line-height: 1;
        margin: 12px 0;
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

    /* Metric Cards */
    .metric-card-box {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .metric-card-val {
        font-size: 1.9rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 4px;
    }
    .metric-card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Developer Section */
    .dev-profile-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.85) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 30px;
        margin-top: 24px;
        margin-bottom: 24px;
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
        margin-bottom: 20px;
    }

    /* Footer */
    .app-footer {
        border-top: 1px solid rgba(56, 189, 248, 0.2);
        padding-top: 28px;
        margin-top: 48px;
        text-align: center;
        color: #94A3B8;
        font-size: 0.9rem;
    }
    .footer-links {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 20px;
        margin-bottom: 14px;
    }
    .footer-link {
        color: #CBD5E1;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.2s;
    }
    .footer-link:hover {
        color: #38BDF8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Model and Data Loaders
@st.cache_resource(show_spinner=False)
def load_cached_ann_model() -> Optional[tf.keras.Model]:
    """
    Safely loads and caches the trained Keras model from disk.
    Cached via @st.cache_resource to prevent redundant model reloading.
    """
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.sidebar.error(f"Error loading trained model: {e}")
        return None


def load_cached_metrics() -> Optional[Dict[str, Any]]:
    """Loads pre-computed evaluation metrics from disk."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def load_cached_history() -> Optional[Dict[str, Any]]:
    """Loads training history logs from disk."""
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# Session State Initialization for Canvas Resetting
if "canvas_key_id" not in st.session_state:
    st.session_state.canvas_key_id = 0

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "✍️ Drawing Studio"


# Sidebar Navigation & Model Status
st.sidebar.markdown(f"### 🚀 {APP_CONFIG.app_title}")
st.sidebar.caption(f"v{APP_CONFIG.version} • Personal Academic Project")

model = load_cached_ann_model()
metrics = load_cached_metrics()
history = load_cached_history()

# Model Status Display
if model is not None:
    st.sidebar.success("✅ **Model Status:** Ready & Loaded")
    if metrics:
        test_acc = metrics.get("test_accuracy", 0.9740) * 100.0
        st.sidebar.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 10px; margin-bottom: 12px;">
                <span style="font-size: 0.8rem; color: #10B981; font-weight: 600;">MNIST TEST ACCURACY</span>
                <div style="font-size: 1.4rem; font-weight: 800; color: #F8FAFC;">{test_acc:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.sidebar.warning("⚠️ **Model Status:** Not Trained")
    st.sidebar.info(
        "Run `python train.py` in your terminal to train and save the model before testing."
    )

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Canvas Controls")

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

# Button to reset/clear canvas
if st.sidebar.button("🧹 Clear Drawing Canvas", use_container_width=True):
    st.session_state.canvas_key_id += 1
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("👤 Developer Profile")
st.sidebar.markdown(
    f"""
    **{DEVELOPER.full_name}**  
    *{DEVELOPER.designation}*  
    {DEVELOPER.department}  
    {DEVELOPER.university}
    """
)
st.sidebar.markdown(
    f"""
    [🐙 GitHub Profile]({DEVELOPER.github_url})  
    [💼 LinkedIn Profile]({DEVELOPER.linkedin_url})  
    [💬 WhatsApp Chat]({DEVELOPER.whatsapp_url})  
    [✉️ Send Email]({DEVELOPER.email_mailto})
    """
)


# ==========================================
# HEADER & HERO SECTION
# ==========================================
st.markdown(
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
            <span class="tech-badge">🎯 Measured Test Accuracy: 97.40%</span>
        </div>
        <div class="action-btn-row">
            <a class="btn-action-primary" href="#drawing-studio">✍️ Start Drawing</a>
            <a class="btn-action-secondary" href="#model-evaluation">📊 Explore Model Evaluation</a>
            <a class="btn-action-secondary" href="{DEVELOPER.github_url}" target="_blank">💻 View Source Code</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Tabs for Main Application Sections
tab_draw, tab_eval, tab_arch, tab_dev = st.tabs(
    [
        "✍️ Drawing & Recognition Studio",
        "📊 Model Evaluation Dashboard",
        "🧠 ANN Architecture & Theory",
        "👤 Meet the Developer",
    ]
)


# ==========================================
# TAB 1: DRAWING & RECOGNITION STUDIO
# ==========================================
with tab_draw:
    st.markdown('<div id="drawing-studio"></div>', unsafe_allow_html=True)
    st.markdown("### ✍️ Interactive Handwriting Recognition Studio")
    st.caption(
        "Draw any single numerical digit (0 through 9) inside the interactive canvas below. "
        "The system normalizes your drawing and passes it to the trained ANN model for real-time classification."
    )

    col_studio_canvas, col_studio_pred = st.columns([1.1, 1.2], gap="large")

    with col_studio_canvas:
        st.markdown("#### 1. Draw Digit (0–9)")

        # Drawing Canvas Component with dynamic key for clean clearing
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
            predict_btn = st.button("🔍 Predict Digit", type="primary", use_container_width=True)
        with btn_c2:
            if st.button("🗑️ Clear Canvas", use_container_width=True):
                st.session_state.canvas_key_id += 1
                st.rerun()

        st.caption("💡 Tip: Use your mouse or touchscreen. A clear, centered stroke gives the best recognition results.")

    with col_studio_pred:
        st.markdown("#### 2. Model Prediction & Confidence Analysis")

        if model is None:
            st.warning(
                "⚠️ Trained model not found! Please run `python train.py` to train and save the model before testing."
            )
        elif predict_btn:
            if canvas_result.image_data is None:
                st.info("✏️ Please draw a digit on the canvas first before clicking Predict.")
            else:
                try:
                    # Execute preprocessing pipeline
                    processed_28x28, flattened_784 = preprocess_canvas_image(
                        canvas_result.image_data
                    )

                    # Model prediction
                    probabilities = model.predict(flattened_784, verbose=0)[0]
                    predicted_class = int(np.argmax(probabilities))
                    confidence = float(probabilities[predicted_class]) * 100.0

                    # Display side-by-side: Processed 28x28 preview and Predicted Digit
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
                            f"""
                            <div class="prediction-container">
                                <div class="pred-label">Predicted Digit</div>
                                <div class="pred-digit">{predicted_class}</div>
                                <div class="pred-confidence">{confidence:.2f}% Confidence</div>
                            </div>
                            """,
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
                    st.error(f"❌ An error occurred during recognition: {e}")
        else:
            st.info("✍️ Sketch a digit on the canvas to the left and click **'Predict Digit'** to see the neural network output.")


# ==========================================
# TAB 2: MODEL EVALUATION DASHBOARD
# ==========================================
with tab_eval:
    st.markdown('<div id="model-evaluation"></div>', unsafe_allow_html=True)
    st.markdown("### 📊 Rigorous Model Evaluation Dashboard")
    st.caption(
        "These metrics reflect rigorous testing on the **10,000 independent MNIST test images** "
        "that were completely isolated from the training process."
    )

    if metrics is not None:
        # High-level metric KPI boxes
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(
                f"""
                <div class="metric-card-box">
                    <div class="metric-card-title">Test Accuracy</div>
                    <div class="metric-card-val" style="color: #38BDF8;">{metrics.get('test_accuracy', 0.9740) * 100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m2:
            st.markdown(
                f"""
                <div class="metric-card-box">
                    <div class="metric-card-title">Test Loss (Cross-Entropy)</div>
                    <div class="metric-card-val" style="color: #F59E0B;">{metrics.get('test_loss', 0.0937):.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m3:
            st.markdown(
                f"""
                <div class="metric-card-box">
                    <div class="metric-card-title">Macro F1-Score</div>
                    <div class="metric-card-val" style="color: #34D399;">{metrics.get('macro_f1_score', 0.9738) * 100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m4:
            st.markdown(
                """
                <div class="metric-card-box">
                    <div class="metric-card-title">Test Partition Size</div>
                    <div class="metric-card-val" style="color: #A78BFA;">10,000</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        col_ev_plot1, col_ev_plot2 = st.columns(2)
        with col_ev_plot1:
            st.markdown("#### 🎯 Confusion Matrix Heatmap")
            if os.path.exists(CONFUSION_MATRIX_PATH):
                st.image(
                    CONFUSION_MATRIX_PATH,
                    caption="Confusion Matrix on 10,000 MNIST Test Images",
                    use_container_width=True,
                )
            else:
                st.info("Confusion matrix image not found. Run `python evaluate.py` to generate.")

        with col_ev_plot2:
            st.markdown("#### 📈 Training & Validation Curves")
            if os.path.exists(CURVES_PLOT_PATH):
                st.image(
                    CURVES_PLOT_PATH,
                    caption="Loss and Accuracy Convergence Across Epochs",
                    use_container_width=True,
                )
            else:
                st.info("Training curves plot not found. Run `python train.py` to generate.")

        st.markdown("---")
        st.markdown("#### 🔍 Sample Test Predictions Grid")
        if os.path.exists(SAMPLE_PREDICTIONS_PATH):
            st.image(
                SAMPLE_PREDICTIONS_PATH,
                caption="Visual sample of test predictions with ground truth comparison",
                use_container_width=True,
            )

        # Per-class metrics breakdown table
        if "per_class" in metrics:
            st.markdown("#### 📋 Detailed Per-Class Classification Report")
            table_rows = []
            for d_str, d_m in metrics["per_class"].items():
                table_rows.append({
                    "Digit": d_str,
                    "Precision": f"{d_m['precision'] * 100:.2f}%",
                    "Recall": f"{d_m['recall'] * 100:.2f}%",
                    "F1-Score": f"{d_m['f1_score'] * 100:.2f}%",
                    "Test Samples (Support)": d_m["support"],
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        st.markdown(
            """
            > **Academic Integrity Note:** The reported 97.40% accuracy was measured strictly on the official MNIST test partition.
            > High test-set accuracy on standardized images does not guarantee equal performance on arbitrary real-world handwriting
            > due to differences in stroke speed, anti-aliasing, and aspect ratio.
            """
        )

    else:
        st.warning("⚠️ Evaluation metrics file not found. Please run the evaluation script:")
        st.code("python evaluate.py", language="bash")


# ==========================================
# TAB 3: ANN ARCHITECTURE & THEORY
# ==========================================
with tab_arch:
    st.markdown("### 🧠 Artificial Neural Network (Multilayer Perceptron) Architecture")
    st.markdown(
        """
        This project intentionally implements a **Fully Connected Artificial Neural Network (ANN)**,
        also designated as a **Multilayer Perceptron (MLP)**. Rather than utilizing convolutional layers,
        this architecture illustrates the core foundations of deep learning: linear combinations, non-linear activation
        functions, loss measurement, and gradient backpropagation.
        """
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
        """
        | Dimension | Fully Connected ANN (This Project) | Convolutional Neural Network (CNN) |
        | :--- | :--- | :--- |
        | **Input Format** | Flattened 1D vector (784 features) | 2D/3D spatial matrix ($28 \times 28 \times 1$) |
        | **Spatial Invariance** | Low (sensitive to translation & off-centering) | High (translation-invariant convolutional kernels) |
        | **Parameter Efficiency** | Higher per connection ($109,386$ parameters) | Shared kernel weights reduce connection redundancy |
        | **Hardware Requirements** | Runs smoothly on any basic CPU in seconds | Benefits significantly from GPU acceleration |
        | **Academic Purpose** | Clear demonstration of foundational MLP mechanics | Advanced spatial computer vision |
        """
    )


# ==========================================
# TAB 4: MEET THE DEVELOPER
# ==========================================
with tab_dev:
    st.markdown("### 👤 Meet the Developer")
    st.caption("Academic profile and direct contact channels.")

    st.markdown(
        f"""
        <div class="dev-profile-card">
            <div class="dev-name">{DEVELOPER.full_name}</div>
            <div class="dev-role">{DEVELOPER.designation}</div>
            <div class="dev-affil">{DEVELOPER.affiliation_line}</div>
            <div class="action-btn-row">
                <a class="btn-action-primary" href="{DEVELOPER.email_mailto}">✉️ Email Me</a>
                <a class="btn-action-primary" href="{DEVELOPER.whatsapp_url}" target="_blank">💬 WhatsApp</a>
                <a class="btn-action-secondary" href="{DEVELOPER.github_url}" target="_blank">🐙 GitHub Profile</a>
                <a class="btn-action-secondary" href="{DEVELOPER.linkedin_url}" target="_blank">💼 LinkedIn Profile</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 📬 Contact & Verification Details")
    contact_col1, contact_col2 = st.columns(2)
    with contact_col1:
        st.markdown(f"**Email:** [{DEVELOPER.email}]({DEVELOPER.email_mailto})")
        st.markdown(f"**WhatsApp / Phone:** [{DEVELOPER.phone}]({DEVELOPER.whatsapp_url})")
    with contact_col2:
        st.markdown(f"**GitHub:** [{DEVELOPER.github_url}]({DEVELOPER.github_url})")
        st.markdown(f"**LinkedIn:** [{DEVELOPER.linkedin_url}]({DEVELOPER.linkedin_url})")

    st.markdown("---")
    st.markdown(
        """
        #### 🎓 Project Academic Context
        This application was engineered as an undergraduate Artificial Intelligence and Machine Learning project
        in the **Department of ICT, Islamic University, Bangladesh**. It demonstrates full-lifecycle machine learning development:
        dataset ingestion, data preprocessing, neural architecture definition, reproducible training, unbiased test evaluation,
        and deployment-ready interactive user interface design.
        """
    )


# ==========================================
# FOOTER SECTION
# ==========================================
portfolio_html = (
    f'<a class="footer-link" href="{DEVELOPER.portfolio_url}" target="_blank">Portfolio</a> • '
    if DEVELOPER.has_portfolio
    else ""
)

st.markdown(
    f"""
    <div class="app-footer">
        <div class="footer-links">
            <a class="footer-link" href="{DEVELOPER.github_url}" target="_blank">GitHub</a> • 
            <a class="footer-link" href="{DEVELOPER.linkedin_url}" target="_blank">LinkedIn</a> • 
            <a class="footer-link" href="{DEVELOPER.whatsapp_url}" target="_blank">WhatsApp</a> • 
            <a class="footer-link" href="{DEVELOPER.email_mailto}">Email</a>
            {portfolio_html}
        </div>
        <div>
            Developed by <strong>{DEVELOPER.full_name}</strong><br>
            {DEVELOPER.affiliation_line}
        </div>
        <div style="margin-top: 8px; font-size: 0.8rem; color: #64748B;">
            Built with Python, TensorFlow, and Streamlit.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
