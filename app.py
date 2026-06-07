"""
================================================================================
  NAFLD Detection System — Streamlit Dashboard
  ML Framework for Non-Alcoholic Fatty Liver Disease Detection
  Author : Your Name
  Purpose: Academic / Educational demonstration only
================================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be the very first call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NAFLD Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS — refined medical-dark aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- Google Font Import ---------- */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ---------- Root Variables ---------- */
:root {
    --bg-primary:   #0d1117;
    --bg-card:      #161b22;
    --bg-card2:     #1c2330;
    --accent-teal:  #00c9b1;
    --accent-amber: #f4a261;
    --accent-red:   #e63946;
    --accent-green: #2ec4b6;
    --text-primary: #e6edf3;
    --text-muted:   #8b949e;
    --border:       rgba(255,255,255,0.07);
    --radius:       14px;
    --shadow:       0 8px 32px rgba(0,0,0,0.45);
}

/* ---------- Global Reset ---------- */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ---------- Hide Streamlit chrome ---------- */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1280px; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ---------- Metric boxes ---------- */
[data-testid="metric-container"] {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
}

/* ---------- Inputs ---------- */
.stSlider > div > div > div > div { background: var(--accent-teal) !important; }
input[type="number"], .stSelectbox select {
    background: var(--bg-card2) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-teal), #009e8c) !important;
    color: #0d1117 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    letter-spacing: 0.04em !important;
    transition: transform .15s, box-shadow .15s !important;
    box-shadow: 0 4px 20px rgba(0,201,177,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(0,201,177,0.4) !important;
}

/* ---------- Utility card class ---------- */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: var(--shadow);
}
.card-accent { border-left: 4px solid var(--accent-teal) !important; }

/* ---------- Risk banners ---------- */
.risk-high {
    background: linear-gradient(135deg, rgba(230,57,70,0.18), rgba(230,57,70,0.08));
    border: 1px solid rgba(230,57,70,0.45);
    border-left: 5px solid var(--accent-red);
    border-radius: var(--radius);
    padding: 1.4rem 1.8rem;
    margin: 1rem 0;
}
.risk-low {
    background: linear-gradient(135deg, rgba(46,196,182,0.18), rgba(46,196,182,0.08));
    border: 1px solid rgba(46,196,182,0.45);
    border-left: 5px solid var(--accent-green);
    border-radius: var(--radius);
    padding: 1.4rem 1.8rem;
    margin: 1rem 0;
}
.risk-title { font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:800; margin:0 0 .35rem 0; }
.risk-sub   { font-size:.92rem; color:var(--text-muted); margin:0; }

/* ---------- Section headings ---------- */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent-teal);
    letter-spacing: .07em;
    text-transform: uppercase;
    margin: 1.6rem 0 .8rem 0;
    border-bottom: 1px solid var(--border);
    padding-bottom: .45rem;
}

/* ---------- Footer ---------- */
.footer {
    margin-top: 3rem;
    padding: 1.2rem 1.8rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    font-size: .8rem;
    color: var(--text-muted);
    text-align: center;
    line-height: 1.7;
}

/* ---------- Hero header ---------- */
.hero {
    background: linear-gradient(135deg, #161b22 0%, #1a2535 60%, #0d1f1d 100%);
    border: 1px solid var(--border);
    border-bottom: 3px solid var(--accent-teal);
    border-radius: var(--radius);
    padding: 2rem 2.4rem 1.6rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(0,201,177,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    margin: 0 0 .3rem 0;
    background: linear-gradient(90deg, var(--accent-teal), #7fffd4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1rem;
    color: var(--text-muted);
    margin: 0;
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,201,177,0.12);
    border: 1px solid rgba(0,201,177,0.3);
    color: var(--accent-teal);
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .1em;
    text-transform: uppercase;
    padding: .25rem .75rem;
    border-radius: 20px;
    margin-bottom: .9rem;
}

/* ---------- Summary table ---------- */
.summary-table { width:100%; border-collapse:collapse; font-size:.88rem; }
.summary-table th {
    background: rgba(0,201,177,0.10);
    color: var(--accent-teal);
    font-weight:600;
    padding:.55rem .9rem;
    text-align:left;
    border-bottom:1px solid var(--border);
    font-size:.78rem;
    letter-spacing:.06em;
    text-transform:uppercase;
}
.summary-table td { padding:.5rem .9rem; border-bottom:1px solid var(--border); }
.summary-table tr:last-child td { border-bottom:none; }
.summary-table tr:hover td { background:rgba(255,255,255,0.025); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CONSTANTS  — exact features used at training
# ─────────────────────────────────────────────
# Replace this list with the EXACT column order your model was trained on.
# Any feature NOT collected via the form is filled with the default/mean below.
TRAINING_FEATURES = [
    "Age", "Sex", "BMI", "Waist_Circumference",
    "Fasting_Blood_Sugar", "Triglycerides", "Total_Cholesterol",
    "Insulin", "CRP", "CAP_Score", "Fibroscan_kPa",
    "ALT", "AST", "GGT",
]

# Default / mean values for features not explicitly entered
FEATURE_DEFAULTS = {
    "Age": 40,
    "Sex": 1,
    "BMI": 25.0,
    "Waist_Circumference": 88.0,
    "Fasting_Blood_Sugar": 95.0,
    "Triglycerides": 130.0,
    "Total_Cholesterol": 190.0,
    "Insulin": 12.0,
    "CRP": 2.5,
    "CAP_Score": 220.0,
    "Fibroscan_kPa": 5.5,
    "ALT": 28.0,
    "AST": 25.0,
    "GGT": 30.0,
}

MODEL_PATH  = "best_model_NAFLD.joblib"
SCALER_PATH = "scaler_NAFLD.joblib"


# ─────────────────────────────────────────────
#  MODEL LOADER  (cached for performance)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load saved model and scaler from disk. Returns (model, scaler, loaded:bool)."""
    model, scaler, loaded = None, None, False
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            model  = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            loaded = True
        except Exception as e:
            st.error(f"⚠️ Error loading model artifacts: {e}")
    return model, scaler, loaded


# ─────────────────────────────────────────────
#  HELPER — build aligned feature vector
# ─────────────────────────────────────────────
def build_feature_vector(user_inputs: dict) -> pd.DataFrame:
    """
    Build a single-row DataFrame aligned to TRAINING_FEATURES.
    Fields not present in user_inputs are filled from FEATURE_DEFAULTS.
    """
    row = {}
    for feat in TRAINING_FEATURES:
        row[feat] = user_inputs.get(feat, FEATURE_DEFAULTS.get(feat, 0))
    return pd.DataFrame([row], columns=TRAINING_FEATURES)


# ─────────────────────────────────────────────
#  HELPER — feature importance chart
# ─────────────────────────────────────────────
def plot_feature_importance(model, top_n: int = 10):
    """
    Render a horizontal bar chart of feature importances.
    Supports tree-based models (feature_importances_) and
    linear models (coef_). Falls back gracefully.
    """
    importances = None
    feat_names  = TRAINING_FEATURES

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
    elif hasattr(model, "estimators_"):          # e.g., VotingClassifier
        for est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                importances = est.feature_importances_
                break

    if importances is None:
        return None

    # Sort top-N
    indices = np.argsort(importances)[::-1][:top_n]
    top_feats  = [feat_names[i] for i in indices]
    top_scores = importances[indices]

    # Build human-readable labels
    label_map = {
        "Age": "Age", "Sex": "Sex", "BMI": "BMI",
        "Waist_Circumference": "Waist Circumf.",
        "Fasting_Blood_Sugar": "Fasting Glucose",
        "Triglycerides": "Triglycerides",
        "Total_Cholesterol": "Total Cholesterol",
        "Insulin": "Insulin", "CRP": "CRP",
        "CAP_Score": "CAP Score",
        "Fibroscan_kPa": "Fibroscan (kPa)",
        "ALT": "ALT", "AST": "AST", "GGT": "GGT",
    }
    labels = [label_map.get(f, f) for f in top_feats]

    # Colour gradient teal→amber
    cmap   = plt.cm.get_cmap("cool", len(top_feats))
    colors = [cmap(i / max(len(top_feats) - 1, 1)) for i in range(len(top_feats))]

    fig, ax = plt.subplots(figsize=(7, max(3, len(top_feats) * 0.45)))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#1c2330")

    bars = ax.barh(labels[::-1], top_scores[::-1], color=colors[::-1],
                   height=0.62, edgecolor="none")

    # Value labels
    for bar, val in zip(bars, top_scores[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left",
                color="#8b949e", fontsize=8.5, fontfamily="monospace")

    ax.tick_params(colors="#e6edf3", labelsize=9)
    ax.xaxis.label.set_color("#8b949e")
    ax.set_xlabel("Importance Score", color="#8b949e", fontsize=9)
    ax.set_title("Top Feature Importances", color="#00c9b1",
                 fontsize=11, fontweight="bold", pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d333b")
    ax.xaxis.set_tick_params(color="#2d333b")
    ax.grid(axis="x", color="#2d333b", linestyle="--", linewidth=0.6, alpha=0.6)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
#  HELPER — gauge chart
# ─────────────────────────────────────────────
def plot_gauge(probability: float):
    """Semicircular risk gauge coloured by probability value."""
    fig, ax = plt.subplots(figsize=(5, 2.8), subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")
    ax.axis("off")

    # Background arc
    theta    = np.linspace(np.pi, 0, 200)
    r_outer, r_inner = 1.0, 0.62
    ax.fill_between(np.cos(theta) * r_outer, np.sin(theta) * r_outer,
                    np.cos(theta) * r_inner, np.sin(theta) * r_inner,
                    color="#2d333b", zorder=1)

    # Coloured risk arc
    risk_theta = np.linspace(np.pi, np.pi - probability * np.pi, 200)
    r_col = np.interp(probability, [0, 0.5, 1.0],
                      [[0.2, 0.8, 0.2], [0.8, 0.7, 0.1], [0.9, 0.2, 0.15]])
    colour = "#{:02x}{:02x}{:02x}".format(int(r_col[0]*255),
                                            int(r_col[1]*255),
                                            int(r_col[2]*255))
    ax.fill_between(np.cos(risk_theta) * r_outer, np.sin(risk_theta) * r_outer,
                    np.cos(risk_theta) * r_inner, np.sin(risk_theta) * r_inner,
                    color=colour, alpha=0.9, zorder=2)

    # Needle
    needle_angle = np.pi - probability * np.pi
    ax.annotate("", xy=(0.78 * np.cos(needle_angle), 0.78 * np.sin(needle_angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#e6edf3",
                                lw=2.0, mutation_scale=14))
    ax.add_patch(plt.Circle((0, 0), 0.07, color="#e6edf3", zorder=5))

    # Labels
    ax.text(-1.08, -0.12, "0%",  ha="center", va="center",
            color="#8b949e", fontsize=8.5, fontfamily="monospace")
    ax.text( 1.08, -0.12, "100%", ha="center", va="center",
            color="#8b949e", fontsize=8.5, fontfamily="monospace")
    ax.text(0, -0.25, f"{probability*100:.1f}%",
            ha="center", va="center", color="#e6edf3",
            fontsize=17, fontweight="bold", fontfamily="monospace")
    ax.text(0, -0.46, "Risk Probability",
            ha="center", va="center", color="#8b949e", fontsize=8.5)

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.6, 1.1)
    plt.tight_layout(pad=0)
    return fig


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:.5rem 0 1.2rem;'>
        <span style='font-size:3rem;'>🫁</span>
        <p style='font-family:Syne,sans-serif; font-size:1.1rem;
                  font-weight:800; margin:.4rem 0 .1rem; color:#00c9b1;'>
            NAFLD Detect
        </p>
        <p style='font-size:.75rem; color:#8b949e; margin:0;'>
            ML Clinical Decision Support
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <p class='section-header' style='margin-top:.2rem;'>About This Tool</p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card card-accent' style='font-size:.85rem; line-height:1.7;'>
    This dashboard uses a trained Machine Learning model to estimate the risk of
    <strong>Non-Alcoholic Fatty Liver Disease (NAFLD)</strong> from routine clinical
    and anthropometric measurements.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p class='section-header'>How to Use</p>
    <div style='font-size:.84rem; color:#8b949e; line-height:1.75;'>
    <b style='color:#e6edf3;'>1.</b> Enter patient clinical values in the form.<br>
    <b style='color:#e6edf3;'>2.</b> Click <b style='color:#00c9b1;'>Run Prediction</b>.<br>
    <b style='color:#e6edf3;'>3.</b> Review the risk score, gauge & feature chart.<br>
    <b style='color:#e6edf3;'>4.</b> Use <b style='color:#00c9b1;'>Reset</b> to clear inputs.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <p class='section-header'>Model Info</p>
    <div style='font-size:.82rem; color:#8b949e; line-height:1.75;'>
    • Framework: Scikit-learn<br>
    • Models evaluated: LR, DT, RF, KNN, SVM<br>
    • Best model auto-selected by ROC-AUC<br>
    • Cross-validated · Joblib serialised<br>
    • Target: Binary NAFLD (Steatosis ≥ S1)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style='font-size:.75rem; color:#8b949e; line-height:1.65;
                background:rgba(244,162,97,0.08); border:1px solid rgba(244,162,97,0.25);
                border-radius:10px; padding:.8rem 1rem;'>
    ⚠️ <strong style='color:#f4a261;'>Educational Use Only</strong><br>
    This tool is <em>not</em> a certified medical device and must not replace
    professional clinical judgment.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD ARTIFACTS
# ─────────────────────────────────────────────
model, scaler, model_loaded = load_artifacts()


# ─────────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='hero-badge'>🔬 ML-Powered Clinical Tool</div>
    <h1 class='hero-title'>NAFLD Detection System</h1>
    <p class='hero-sub'>
        Machine Learning-based Liver Disease Risk Prediction &nbsp;·&nbsp;
        Academic Research Framework
    </p>
</div>
""", unsafe_allow_html=True)

# Model status banner
if model_loaded:
    st.success("✅ Model & scaler loaded successfully — ready for prediction.", icon="🤖")
else:
    st.warning(
        "⚠️ `best_model_NAFLD.joblib` or `scaler_NAFLD.joblib` not found in the working "
        "directory. Place them alongside `app.py` and restart. "
        "A **demo / mock prediction** will be shown in the meantime.",
        icon="📂",
    )


# ─────────────────────────────────────────────
#  SESSION STATE initialisation
# ─────────────────────────────────────────────
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "reset_flag" not in st.session_state:
    st.session_state.reset_flag = False


# ─────────────────────────────────────────────
#  INPUT FORM
# ─────────────────────────────────────────────
st.markdown("<p class='section-header'>Patient Clinical Parameters</p>",
            unsafe_allow_html=True)

with st.form(key="prediction_form", clear_on_submit=False):

    # ---------- Row 1: Demographics ----------
    st.markdown("<div class='card card-accent'>", unsafe_allow_html=True)
    st.markdown("**👤 Demographics & Anthropometrics**")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        age = st.slider("Age (years)", 18, 90,
                        value=40 if not st.session_state.reset_flag else 40,
                        help="Patient age in completed years")
    with c2:
        sex_label = st.selectbox("Sex", ["Male", "Female"],
                                 help="Biological sex of patient")
        sex = 1 if sex_label == "Male" else 0

    with c3:
        bmi = st.number_input("BMI (kg/m²)", min_value=14.0, max_value=65.0,
                              value=25.0, step=0.1,
                              help="Body Mass Index = weight(kg)/height²(m)")
    with c4:
        waist = st.number_input("Waist Circumference (cm)", min_value=50.0,
                                max_value=180.0, value=88.0, step=0.5,
                                help="Measured at navel level")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Row 2: Metabolic ----------
    st.markdown("<div class='card card-accent'>", unsafe_allow_html=True)
    st.markdown("**🩸 Metabolic & Biochemical Markers**")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        fbs = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=50.0,
                              max_value=500.0, value=95.0, step=1.0,
                              help="Glucose after 8-hour fast")
    with c2:
        trig = st.number_input("Triglycerides (mg/dL)", min_value=30.0,
                               max_value=1000.0, value=130.0, step=1.0,
                               help="Serum triglyceride level")
    with c3:
        chol = st.number_input("Total Cholesterol (mg/dL)", min_value=50.0,
                               max_value=600.0, value=190.0, step=1.0,
                               help="Total serum cholesterol")
    with c4:
        insulin = st.number_input("Insulin (µIU/mL)", min_value=1.0,
                                  max_value=300.0, value=12.0, step=0.5,
                                  help="Fasting serum insulin")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Row 3: Liver markers ----------
    st.markdown("<div class='card card-accent'>", unsafe_allow_html=True)
    st.markdown("**🫀 Inflammatory & Liver Enzyme Markers**")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        crp = st.number_input("CRP (mg/L)", min_value=0.0, max_value=200.0,
                              value=2.5, step=0.1,
                              help="C-Reactive Protein (inflammation marker)")
    with c2:
        alt = st.number_input("ALT (U/L)", min_value=0.0, max_value=1000.0,
                              value=28.0, step=1.0,
                              help="Alanine Aminotransferase")
    with c3:
        ast = st.number_input("AST (U/L)", min_value=0.0, max_value=1000.0,
                              value=25.0, step=1.0,
                              help="Aspartate Aminotransferase")
    with c4:
        ggt = st.number_input("GGT (U/L)", min_value=0.0, max_value=1000.0,
                              value=30.0, step=1.0,
                              help="Gamma-Glutamyl Transferase")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Row 4: Imaging ----------
    st.markdown("<div class='card card-accent'>", unsafe_allow_html=True)
    st.markdown("**📡 Imaging — FibroScan / Controlled Attenuation Parameter**")
    c1, c2 = st.columns(2)

    with c1:
        cap = st.slider("CAP Score (dB/m)", min_value=100, max_value=400,
                        value=220,
                        help="Controlled Attenuation Parameter — hepatic steatosis proxy")
    with c2:
        fibro = st.number_input("FibroScan LSM (kPa)", min_value=1.0,
                                max_value=75.0, value=5.5, step=0.1,
                                help="Liver Stiffness Measurement (fibrosis)")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Buttons ----------
    col_pred, col_reset, _ = st.columns([1.4, 1, 4])
    with col_pred:
        submit = st.form_submit_button("🔍  Run Prediction", use_container_width=True)
    with col_reset:
        reset = st.form_submit_button("↺  Reset", use_container_width=True)


# ─────────────────────────────────────────────
#  RESET LOGIC
# ─────────────────────────────────────────────
if reset:
    st.session_state.prediction_done = False
    st.session_state.reset_flag = True
    st.rerun()


# ─────────────────────────────────────────────
#  PREDICTION LOGIC
# ─────────────────────────────────────────────
if submit:
    st.session_state.reset_flag = False

    # Collect user inputs
    user_inputs = {
        "Age":               age,
        "Sex":               sex,
        "BMI":               bmi,
        "Waist_Circumference": waist,
        "Fasting_Blood_Sugar": fbs,
        "Triglycerides":     trig,
        "Total_Cholesterol": chol,
        "Insulin":           insulin,
        "CRP":               crp,
        "CAP_Score":         cap,
        "Fibroscan_kPa":     fibro,
        "ALT":               alt,
        "AST":               ast,
        "GGT":               ggt,
    }

    # Build aligned DataFrame
    input_df = build_feature_vector(user_inputs)

    # ── Scale & predict ──────────────────────
    if model_loaded:
        # Fix for the merged-line bug: scale X_test, then compute corr separately
        X_test_scaled = scaler.transform(input_df)   # line 1 (fixed)
        # corr = df.corr()                            # line 2 — not needed here but separated

        prediction  = model.predict(X_test_scaled)[0]
        probability = model.predict_proba(X_test_scaled)[0][1]
    else:
        # Demo fallback when model files are absent
        np.random.seed(42)
        prediction  = np.random.randint(0, 2)
        probability = np.random.uniform(0.2, 0.85)

    # Store results
    st.session_state.prediction_done = True
    st.session_state.prediction       = int(prediction)
    st.session_state.probability      = float(probability)
    st.session_state.input_df         = input_df
    st.session_state.user_inputs      = user_inputs
    st.session_state.sex_label        = sex_label


# ─────────────────────────────────────────────
#  RESULTS PANEL
# ─────────────────────────────────────────────
if st.session_state.prediction_done:
    pred  = st.session_state.prediction
    prob  = st.session_state.probability
    inp   = st.session_state.user_inputs

    st.markdown("<p class='section-header'>Prediction Results</p>",
                unsafe_allow_html=True)

    # ── Risk banner ──────────────────────────
    if pred == 1:
        st.markdown("""
        <div class='risk-high'>
            <p class='risk-title' style='color:#e63946;'>
                ⚠️ High Risk of NAFLD
            </p>
            <p class='risk-sub'>
                The model predicts a <strong>positive NAFLD signal</strong>.
                Please refer this patient to a hepatologist for further clinical evaluation.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='risk-low'>
            <p class='risk-title' style='color:#2ec4b6;'>
                ✅ Low Risk of NAFLD
            </p>
            <p class='risk-sub'>
                The model predicts a <strong>negative NAFLD signal</strong>.
                Continue routine monitoring with lifestyle counselling.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Metrics row ──────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("Prediction",        "NAFLD Positive" if pred else "NAFLD Negative")
    m2.metric("Risk Probability",  f"{prob*100:.1f}%")
    m3.metric("Confidence",        f"{max(prob, 1-prob)*100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gauge + Feature importance ───────────
    col_gauge, col_feat = st.columns([1, 1.6])

    with col_gauge:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**📊 Risk Gauge**")
        gauge_fig = plot_gauge(prob)
        st.pyplot(gauge_fig, use_container_width=True)
        plt.close(gauge_fig)

        # Progress bar alternative
        bar_colour = "#e63946" if prob >= 0.5 else "#2ec4b6"
        st.markdown(f"""
        <div style='margin-top:.8rem;'>
            <p style='font-size:.8rem; color:#8b949e; margin-bottom:.3rem;'>
                Risk Level
            </p>
            <div style='background:#2d333b; border-radius:99px; height:10px; overflow:hidden;'>
                <div style='width:{prob*100:.1f}%; background:{bar_colour};
                            height:100%; border-radius:99px;
                            transition:width .6s ease;'></div>
            </div>
            <div style='display:flex; justify-content:space-between;
                        font-size:.72rem; color:#8b949e; margin-top:.3rem;'>
                <span>Low</span><span>Moderate</span><span>High</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_feat:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**📈 Feature Importance (Top 10)**")
        if model_loaded:
            fig_imp = plot_feature_importance(model, top_n=10)
            if fig_imp:
                st.pyplot(fig_imp, use_container_width=True)
                plt.close(fig_imp)
            else:
                st.info("Feature importance is not available for this model type "
                        "(e.g., SVM with RBF kernel).")
        else:
            # Mock importance for demo
            demo_feat   = TRAINING_FEATURES[:10]
            demo_scores = np.random.dirichlet(np.ones(10))
            fig_demo, ax = plt.subplots(figsize=(7, 4))
            fig_demo.patch.set_facecolor("#161b22")
            ax.set_facecolor("#1c2330")
            ax.barh(demo_feat, demo_scores, color="#00c9b1", alpha=0.75)
            ax.set_title("Feature Importance (Demo)", color="#00c9b1", fontsize=10)
            ax.tick_params(colors="#e6edf3")
            for s in ax.spines.values(): s.set_edgecolor("#2d333b")
            plt.tight_layout()
            st.pyplot(fig_demo, use_container_width=True)
            plt.close(fig_demo)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Input Summary table ───────────────────
    st.markdown("<p class='section-header'>Entered Values Summary</p>",
                unsafe_allow_html=True)

    summary_rows = [
        ("Age",               f"{inp['Age']} years"),
        ("Sex",               st.session_state.sex_label),
        ("BMI",               f"{inp['BMI']:.1f} kg/m²"),
        ("Waist Circumference",  f"{inp['Waist_Circumference']:.1f} cm"),
        ("Fasting Blood Sugar",  f"{inp['Fasting_Blood_Sugar']:.1f} mg/dL"),
        ("Triglycerides",     f"{inp['Triglycerides']:.1f} mg/dL"),
        ("Total Cholesterol", f"{inp['Total_Cholesterol']:.1f} mg/dL"),
        ("Insulin",           f"{inp['Insulin']:.1f} µIU/mL"),
        ("CRP",               f"{inp['CRP']:.1f} mg/L"),
        ("ALT",               f"{inp['ALT']:.1f} U/L"),
        ("AST",               f"{inp['AST']:.1f} U/L"),
        ("GGT",               f"{inp['GGT']:.1f} U/L"),
        ("CAP Score",         f"{inp['CAP_Score']} dB/m"),
        ("FibroScan LSM",     f"{inp['Fibroscan_kPa']:.1f} kPa"),
    ]

    rows_html = "".join(
        f"<tr><td>{name}</td><td><b>{val}</b></td></tr>"
        for name, val in summary_rows
    )

    # Split into two visual columns
    half      = len(summary_rows) // 2
    left_rows = "".join(f"<tr><td>{n}</td><td><b>{v}</b></td></tr>"
                        for n, v in summary_rows[:half])
    right_rows = "".join(f"<tr><td>{n}</td><td><b>{v}</b></td></tr>"
                         for n, v in summary_rows[half:])
    st.markdown(f"""
    <div class='card'>
    <div style='display:grid; grid-template-columns:1fr 1fr; gap:1.5rem;'>
        <table class='summary-table'>
            <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
            <tbody>{left_rows}</tbody>
        </table>
        <table class='summary-table'>
            <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
            <tbody>{right_rows}</tbody>
        </table>
    </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class='footer'>
    <strong>⚕️ Medical Disclaimer</strong><br>
    This tool is developed for <strong>educational and research purposes only</strong>
    and does <strong>not</strong> constitute medical advice, diagnosis, or treatment.<br>
    All predictions are generated by a machine learning model and must be interpreted
    by a qualified healthcare professional.<br><br>
    <span style='font-size:.73rem;'>
        NAFLD Detection System · ML Academic Project ·
        Model: Scikit-learn · Built with Streamlit
    </span>
</div>
""", unsafe_allow_html=True)
