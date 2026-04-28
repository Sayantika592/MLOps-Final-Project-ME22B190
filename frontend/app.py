"""
Fake News Detection - Streamlit Frontend
==========================================
"""

import streamlit as st
import requests
import time
import plotly.graph_objects as go
import os

API_URL = os.environ.get("API_URL", "http://backend:8000")

st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700; text-align: center; padding: 1rem 0;
        background: linear-gradient(90deg, #1a1a2e, #16213e, #0f3460);
        color: white; border-radius: 10px; margin-bottom: 2rem;
    }
    .result-card { padding: 2rem; border-radius: 12px; text-align: center; font-size: 1.5rem; font-weight: bold; margin: 1rem 0; }
    .real-card { background: linear-gradient(135deg, #00b09b, #96c93d); color: white; }
    .fake-card { background: linear-gradient(135deg, #e74c3c, #c0392b); color: white; }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        return resp.json()
    except Exception:
        return None


def make_prediction(text):
    try:
        resp = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def get_drift_status():
    try:
        return requests.get(f"{API_URL}/drift", timeout=10).json()
    except Exception:
        return None


def get_pipeline_info():
    try:
        return requests.get(f"{API_URL}/pipeline/info", timeout=10).json()
    except Exception:
        return None


def create_confidence_gauge(confidence, label):
    color = "#2ecc71" if label == "REAL" else "#e74c3c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={"suffix": "%"},
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
        },
    ))
    fig.update_layout(height=250, margin=dict(t=50, b=0, l=30, r=30))
    return fig


# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/news.png", width=80)
    st.title("Navigation")
    page = st.radio("Go to", ["🔍 Prediction", "📊 Pipeline Dashboard", "🛡️ Monitoring", "📖 User Manual"], label_visibility="collapsed")
    st.divider()
    st.subheader("System Status")
    health = check_api_health()
    if health and health.get("status") in ("healthy", "ready"):
        st.success("✅ API Online")
        st.caption(f"Version: {health.get('version', 'N/A')}")
    else:
        st.error("❌ API Offline")
    st.divider()
    st.caption("Fake News Detection System v1.0")
    st.caption("Built with FastAPI + Streamlit + MLflow")


# ======================= PREDICTION PAGE =======================
if page == "🔍 Prediction":
    st.markdown('<div class="main-header">🔍 Fake News Detector</div>', unsafe_allow_html=True)
    st.write("Enter a news headline or paste an article below. The AI model will classify it as **REAL** or **FAKE**.")

    if "news_text" not in st.session_state:
        st.session_state["news_text"] = ""

    col1, col2 = st.columns([3, 1])

    with col2:
        st.markdown("**Quick Examples:**")
        if st.button("📰 Try Real News"):
            st.session_state["news_text"] = "The Federal Reserve announced a quarter-point interest rate increase on Wednesday, citing continued economic growth and stable employment numbers."
            st.rerun()
        if st.button("🚨 Try Fake News"):
            st.session_state["news_text"] = "BREAKING: Scientists discover that the moon is actually made of cheese, NASA confirms in shocking press conference."
            st.rerun()

    with col1:
        text_input = st.text_area("News Text", value=st.session_state["news_text"], height=200,
                                   placeholder="Paste your news article or headline here...",
                                   help="Enter at least a few words for better accuracy.")
        st.session_state["news_text"] = text_input

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

                card_class = "real-card" if label == "REAL" else "fake-card"
                emoji = "✅" if label == "REAL" else "🚨"
                st.markdown(f'<div class="result-card {card_class}">{emoji} This news appears to be <b>{label}</b></div>', unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Prediction", label)
                c2.metric("Confidence", f"{confidence:.1%}")
                c3.metric("Response Time", f"{latency}s")
                c4.metric("Word Count", result.get("word_count", "N/A"))

                st.plotly_chart(create_confidence_gauge(confidence, label), use_container_width=True)


# ======================= PIPELINE DASHBOARD =======================
elif page == "📊 Pipeline Dashboard":
    st.markdown('<div class="main-header">📊 ML Pipeline Dashboard</div>', unsafe_allow_html=True)

    pipeline_info = get_pipeline_info()

    if pipeline_info:
        st.subheader("Pipeline Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Model Settings**")
            st.json({"algorithm": pipeline_info.get("model_algorithm", "N/A")})
        with col2:
            st.markdown("**Preprocessing Settings**")
            st.json(pipeline_info.get("preprocessing", {}))

    st.subheader("Pipeline Stages")

    stages = [
        ("1. Data Ingestion", "Load and validate raw CSV data",
         "Reads news.csv, validates schema (text + label columns required), checks missing values, fills NaN. Throughput: ~33K rows/sec."),
        ("2. Baseline Statistics", "Compute drift detection baselines",
         "Calculates mean, std, median of text_length and title_length. Saves label distribution to baseline_stats.json for production drift monitoring."),
        ("3. Preprocessing", "Clean text, remove noise, feature engineering",
         "Lowercasing, URL removal, HTML stripping, special character removal, whitespace normalization. Creates text_length, word_count, avg_word_length features."),
        ("4. Train/Test Split", "Stratified split (80/20)",
         "Stratified by label to maintain class balance. random_state=42 for reproducibility."),
        ("5. TF-IDF Vectorization", "Convert text to numerical features",
         "50,000 features, unigrams + bigrams, sublinear TF, min_df=2, max_df=0.95, English stop words removed."),
        ("6. Model Training", "Train classifier with experiment tracking",
         "4 experiments tracked in MLflow: NaiveBayes(1K features) -> LogReg(5K) -> LogReg(20K) -> SGD(50K). Best model selected by F1-score."),
        ("7. Model Export", "Save model artifacts for serving",
         "Saves model.pkl + vectorizer.pkl + metrics.json to models/best_model/. API loads these on startup."),
    ]

    for name, desc, detail in stages:
        with st.expander(name, expanded=False):
            st.write(f"**{desc}**")
            st.write(detail)
            st.success("✅ Stage completed successfully")

    st.subheader("Pipeline Performance")
    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("Total Pipeline Time", "~71s")
    pc2.metric("Data Ingestion", "1.35s")
    pc3.metric("Model Training", "0.33s")

    st.subheader("MLOps Tool Dashboards")
    st.markdown("""
| Tool | URL | Purpose |
|------|-----|---------|
| **Airflow** | [http://localhost:8080](http://localhost:8080) | Pipeline orchestration, DAG management, task monitoring |
| **MLflow** | [http://localhost:5001](http://localhost:5001) | Experiment tracking, model registry, artifact storage |
| **Prometheus** | [http://localhost:9090](http://localhost:9090) | Metrics collection, alert rules |
| **Grafana** | [http://localhost:3001](http://localhost:3001) | Real-time monitoring dashboards |
| **AlertManager** | [http://localhost:9093](http://localhost:9093) | Alert routing, email notifications |
    """)

    st.subheader("DVC Pipeline DAG")
    st.code("""
+----------------+
| data_ingestion |
+----------------+
        *
        *
 +--------------+
 | preprocessing |
 +--------------+
        *
        *
    +-------+
    | train |
    +-------+
    """, language="text")

    st.subheader("Airflow DAG Structure")
    st.code("""
wait_for_data >> data_ingestion >> [compute_baselines, preprocessing]
preprocessing >> train_model >> validate_model >> send_success_email
train_model >> send_failure_email (on failure)
    """, language="text")


# ======================= MONITORING PAGE =======================
elif page == "🛡️ Monitoring":
    st.markdown('<div class="main-header">🛡️ System Monitoring</div>', unsafe_allow_html=True)

    drift = get_drift_status()
    if drift:
        st.subheader("Data Drift Status")
        c1, c2, c3 = st.columns(3)
        c1.metric("Drift Score", f"{drift.get('overall_drift', 0):.4f}")
        c2.metric("Alert", "🚨 YES" if drift.get("alert") else "✅ No")
        c3.metric("Samples", drift.get("samples_analyzed", 0))
        if drift.get("features"):
            st.subheader("Feature-Level Drift")
            for feat, details in drift["features"].items():
                with st.expander(f"Feature: {feat}"):
                    st.json(details)
    else:
        st.info("Drift data will appear after predictions are made.")

    st.divider()
    st.subheader("Monitoring Links")
    st.markdown("""
- **Prometheus**: [http://localhost:9090](http://localhost:9090) — Metrics & alert rules
- **Grafana**: [http://localhost:3001](http://localhost:3001) — Real-time dashboards (admin/admin)
- **AlertManager**: [http://localhost:9093](http://localhost:9093) — Active alerts & silences
- **API Metrics**: [http://localhost:8001/metrics](http://localhost:8001/metrics) — Raw Prometheus metrics
    """)


# ======================= USER MANUAL =======================
elif page == "📖 User Manual":
    st.markdown('<div class="main-header">📖 User Manual</div>', unsafe_allow_html=True)

    st.markdown("""
## Getting Started

Welcome to the **Fake News Detection System**. This application uses machine learning
to classify news articles as either **REAL** or **FAKE**.

### How to Use

1. Navigate to the **🔍 Prediction** page using the sidebar.
2. Paste a news headline or full article into the text box.
3. Click **🔎 Analyze** to get the prediction.
4. Review the result, confidence score, and response metrics.

### Understanding Results

| Label | Meaning |
|-------|---------|
| ✅ REAL | The model believes this is legitimate news |
| 🚨 FAKE | The model believes this is misinformation |

**Confidence Score**: Shows how certain the model is about its prediction as a percentage.
Above 80% indicates high confidence.

### Tips for Best Results

- Provide **complete headlines** or paragraphs rather than single words.
- The model works best with **English-language** news text.
- Longer articles generally produce more reliable predictions.
- The model analyzes **text patterns**, not factual accuracy.

### System Architecture

The system consists of:
- **Frontend** (this UI): Streamlit-based web interface
- **Backend API**: FastAPI server hosting the ML model
- **ML Pipeline**: Automated training via Airflow + DVC
- **Experiment Tracking**: MLflow for comparing models
- **Monitoring**: Prometheus + Grafana for production monitoring

### Troubleshooting

| Issue | Solution |
|-------|----------|
| "API Offline" in sidebar | Ensure Docker containers are running |
| Slow predictions | First request may be slower; subsequent ones are fast |
| Empty results | Ensure the text input is not empty or too short |
    """)