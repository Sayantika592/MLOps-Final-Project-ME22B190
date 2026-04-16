"""
Fake News Detection — Streamlit Frontend
==========================================
A user-friendly web interface for classifying news articles as REAL or FAKE
and visualizing the ML pipeline status and monitoring metrics.
"""

import streamlit as st
import requests
import time
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL = "http://backend:8000"  # Docker service name; change to localhost for local dev
# Fallback for local development
import os
if os.environ.get("API_URL"):
    API_URL = os.environ["API_URL"]

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1a1a2e, #16213e, #0f3460);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .result-card {
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .real-card {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
    }
    .fake-card {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
    }
    .metric-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def check_api_health():
    """Check if the backend API is reachable."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        return resp.json()
    except Exception:
        return None


def make_prediction(text: str) -> dict:
    """Send a prediction request to the API."""
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            json={"text": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def get_drift_status() -> dict:
    """Fetch drift detection status."""
    try:
        resp = requests.get(f"{API_URL}/drift", timeout=10)
        return resp.json()
    except Exception:
        return None


def get_pipeline_info() -> dict:
    """Fetch pipeline configuration info."""
    try:
        resp = requests.get(f"{API_URL}/pipeline/info", timeout=10)
        return resp.json()
    except Exception:
        return None


def create_confidence_gauge(confidence: float, label: str):
    """Create a Plotly gauge chart for prediction confidence."""
    color = "#2ecc71" if label == "REAL" else "#e74c3c"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100 if confidence <= 1.0 else confidence,
        title={"text": "Confidence Score", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 33], "color": "#fadbd8"},
                {"range": [33, 66], "color": "#fdebd0"},
                {"range": [66, 100], "color": "#d5f5e3"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 2},
                "thickness": 0.75,
                "value": confidence * 100 if confidence <= 1.0 else confidence,
            },
        },
    ))
    fig.update_layout(height=250, margin=dict(t=50, b=0, l=30, r=30))
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/news.png", width=80)
    st.title("Navigation")

    page = st.radio(
        "Go to",
        ["🔍 Prediction", "📊 Pipeline Dashboard", "🛡️ Monitoring", "📖 User Manual"],
        label_visibility="collapsed",
    )

    st.divider()

    # API Health Status
    st.subheader("System Status")
    health = check_api_health()
    if health and health.get("status") in ("healthy", "ready"):
        st.success("✅ API Online")
        st.caption(f"Version: {health.get('version', 'N/A')}")
    else:
        st.error("❌ API Offline")
        st.caption("Ensure the backend container is running.")

    st.divider()
    st.caption("Fake News Detection System v1.0")
    st.caption("Built with FastAPI + Streamlit + MLflow")


# ---------------------------------------------------------------------------
# Page: Prediction
# ---------------------------------------------------------------------------

if page == "🔍 Prediction":
    st.markdown('<div class="main-header">🔍 Fake News Detector</div>', unsafe_allow_html=True)
    st.markdown(
        "Enter a news headline or paste an article below. "
        "The AI model will classify it as **REAL** or **FAKE**."
    )

    # Input area
    col1, col2 = st.columns([3, 1])

    with col1:
        text_input = st.text_area(
            "News Text",
            height=200,
            placeholder="Paste your news article or headline here...",
            help="Enter at least a few words for better accuracy.",
        )

    with col2:
        st.markdown("**Quick Examples:**")
        if st.button("📰 Try Real News"):
            text_input = "The Federal Reserve announced a quarter-point interest rate increase on Wednesday, citing continued economic growth and stable employment numbers."
            st.session_state["example_text"] = text_input
        if st.button("🚨 Try Fake News"):
            text_input = "BREAKING: Scientists discover that the moon is actually made of cheese, NASA confirms in shocking press conference."
            st.session_state["example_text"] = text_input

    # Use example text if set
    if "example_text" in st.session_state:
        text_input = st.session_state.pop("example_text")

    # Predict button
    if st.button("🔎 Analyze", type="primary", use_container_width=True):
        if not text_input or len(text_input.strip()) < 5:
            st.warning("⚠️ Please enter a longer text for analysis.")
        else:
            with st.spinner("Analyzing article..."):
                start = time.time()
                result = make_prediction(text_input)
                latency = round(time.time() - start, 3)

            if "error" in result:
                st.error(f"❌ Error: {result['error']}")
            else:
                label = result["label"]
                confidence = result["confidence"]

                # Result card
                card_class = "real-card" if label == "REAL" else "fake-card"
                emoji = "✅" if label == "REAL" else "🚨"
                st.markdown(
                    f'<div class="result-card {card_class}">{emoji} This news appears to be <b>{label}</b></div>',
                    unsafe_allow_html=True,
                )

                # Metrics row
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Prediction", label)
                c2.metric("Confidence", f"{confidence:.2%}" if confidence <= 1.0 else f"{confidence:.2f}")
                c3.metric("Response Time", f"{latency}s")
                c4.metric("Word Count", result.get("word_count", "N/A"))

                # Confidence gauge
                st.plotly_chart(
                    create_confidence_gauge(confidence, label),
                    use_container_width=True,
                )


# ---------------------------------------------------------------------------
# Page: Pipeline Dashboard
# ---------------------------------------------------------------------------

elif page == "📊 Pipeline Dashboard":
    st.markdown('<div class="main-header">📊 ML Pipeline Dashboard</div>', unsafe_allow_html=True)

    pipeline_info = get_pipeline_info()

    if pipeline_info:
        st.subheader("Pipeline Configuration")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Model Settings**")
            st.json({
                "algorithm": pipeline_info.get("model_algorithm", "N/A"),
            })

        with col2:
            st.markdown("**Preprocessing Settings**")
            st.json(pipeline_info.get("preprocessing", {}))

    # Pipeline stages visualization
    st.subheader("Pipeline Stages")

    stages = [
        ("1. Data Ingestion", "Load and validate raw CSV data"),
        ("2. Baseline Statistics", "Compute drift detection baselines"),
        ("3. Preprocessing", "Clean text, remove noise, feature engineering"),
        ("4. Train/Test Split", "Stratified split (80/20)"),
        ("5. TF-IDF Vectorization", "Convert text to numerical features"),
        ("6. Model Training", "Train classifier with experiment tracking"),
        ("7. Model Export", "Save model artifacts for serving"),
    ]

    for name, desc in stages:
        with st.expander(name, expanded=False):
            st.write(desc)
            st.success("Stage defined in pipeline configuration.")

    # MLflow link
    st.subheader("Experiment Tracking")
    st.info("📈 MLflow UI is available at http://localhost:5000 when running locally.")


# ---------------------------------------------------------------------------
# Page: Monitoring
# ---------------------------------------------------------------------------

elif page == "🛡️ Monitoring":
    st.markdown('<div class="main-header">🛡️ System Monitoring</div>', unsafe_allow_html=True)

    drift = get_drift_status()

    if drift:
        st.subheader("Data Drift Status")

        col1, col2, col3 = st.columns(3)
        col1.metric("Drift Score", f"{drift.get('overall_drift', 0):.4f}")
        col2.metric("Alert", "🚨 YES" if drift.get("alert") else "✅ No")
        col3.metric("Samples", drift.get("samples_analyzed", 0))

        if drift.get("features"):
            st.subheader("Feature-Level Drift")
            for feat, details in drift["features"].items():
                with st.expander(f"Feature: {feat}"):
                    st.json(details)
    else:
        st.info("Drift data will appear after predictions are made.")

    st.divider()
    st.subheader("Prometheus & Grafana")
    st.markdown(
        "- **Prometheus**: http://localhost:9090\n"
        "- **Grafana**: http://localhost:3000 (admin/admin)\n"
        "- **API Metrics**: Available at `/metrics` endpoint"
    )


# ---------------------------------------------------------------------------
# Page: User Manual
# ---------------------------------------------------------------------------

elif page == "📖 User Manual":
    st.markdown('<div class="main-header">📖 User Manual</div>', unsafe_allow_html=True)

    st.markdown("""
    ## Getting Started

    Welcome to the **Fake News Detection System**. This application uses machine learning
    to classify news articles as either **REAL** or **FAKE**.

    ### How to Use

    1. **Navigate** to the **🔍 Prediction** page using the sidebar.
    2. **Paste** a news headline or full article into the text box.
    3. Click **🔎 Analyze** to get the prediction.
    4. Review the result, confidence score, and response metrics.

    ### Understanding Results

    | Label | Meaning |
    |-------|---------|
    | ✅ REAL | The model believes this is legitimate news |
    | 🚨 FAKE | The model believes this is misinformation |

    **Confidence Score**: Higher values indicate the model is more certain.
    A score above 70% generally means high confidence.

    ### Tips for Best Results

    - Provide **complete headlines** or paragraphs rather than single words.
    - The model works best with **English-language** news text.
    - Longer articles generally produce more reliable predictions.
    - The model analyzes **text patterns**, not factual accuracy.

    ### System Architecture

    The system consists of:
    - **Frontend** (this UI): Streamlit-based web interface
    - **Backend API**: FastAPI server hosting the ML model
    - **ML Pipeline**: Automated training pipeline with experiment tracking
    - **Monitoring**: Prometheus + Grafana for production monitoring

    ### Troubleshooting

    | Issue | Solution |
    |-------|----------|
    | "API Offline" in sidebar | Ensure Docker containers are running |
    | Slow predictions | Check system resources; the first request may be slower |
    | Empty results | Ensure the text input is not empty or too short |

    ### Technical Details

    - **Model**: PassiveAggressiveClassifier with TF-IDF features
    - **Training Data**: Labeled fake/real news dataset
    - **API Framework**: FastAPI with automatic OpenAPI docs at `/docs`
    - **Containerization**: Docker + Docker Compose
    """)
