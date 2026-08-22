import warnings
warnings.filterwarnings("ignore")

import os
import textwrap
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)

from catboost import CatBoostClassifier


st.set_page_config(
    page_title="CareerAI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)


PLOT_BG = "#0b1020"
PLOT_FG = "#ffffff"
PLOT_GRID = "#2a2440"
PLOT_C1 = "#a855f7"
PLOT_C2 = "#f472b6"
PLOT_C3 = "#fbbf24"
PLOT_C4 = "#34d399"

warm_cmap = mcolors.LinearSegmentedColormap.from_list(
    "careerai_warm", ["#150f26", "#6d28d9", "#c026d3", "#f472b6", "#fbbf24"]
)

mpl.rcParams.update({
    "figure.facecolor": PLOT_BG,
    "axes.facecolor": PLOT_BG,
    "savefig.facecolor": PLOT_BG,
    "axes.edgecolor": PLOT_GRID,
    "axes.labelcolor": PLOT_FG,
    "text.color": PLOT_FG,
    "xtick.color": PLOT_FG,
    "ytick.color": PLOT_FG,
    "grid.color": PLOT_GRID,
    "axes.grid": True,
    "grid.alpha": 0.30,
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.titlecolor": PLOT_FG,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.labelcolor": PLOT_FG,
})

sns.set_style("darkgrid", {
    "axes.facecolor": PLOT_BG,
    "figure.facecolor": PLOT_BG,
    "grid.color": PLOT_GRID,
})


def render_html(markup):
    st.markdown(textwrap.dedent(markup).strip("\n"), unsafe_allow_html=True)


render_html(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    :root {
        --bg-1: #0a0714;
        --bg-2: #120b24;
        --bg-3: #1a0f2e;
        --ink-0: #ffffff;
        --ink-1: #e4dcf5;
        --ink-2: #b3a6d1;
        --accent-1: #a855f7;
        --accent-2: #f472b6;
        --accent-3: #fbbf24;
        --border-soft: rgba(255,255,255,0.10);
        --border-strong: rgba(244,114,182,0.40);
        --radius-lg: 24px;
        --radius-md: 18px;
        --radius-sm: 12px;
        --shadow-lg: 0 25px 70px rgba(0,0,0,0.40);
        --shadow-md: 0 15px 40px rgba(0,0,0,0.22);
    }
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp {
        background:
            radial-gradient(circle at 8% 8%,  rgba(168,85,247,0.22), transparent 34%),
            radial-gradient(circle at 92% 15%, rgba(244,114,182,0.18), transparent 34%),
            radial-gradient(circle at 50% 100%, rgba(251,191,36,0.10), transparent 42%),
            linear-gradient(160deg, var(--bg-1) 0%, var(--bg-2) 50%, var(--bg-3) 100%);
        background-size: 140% 140%;
        animation: gradientShift 24s ease-in-out infinite;
        color: var(--ink-0);
    }
    @keyframes gradientShift {
        0%   { background-position: 0% 0%; }
        50%  { background-position: 100% 60%; }
        100% { background-position: 0% 0%; }
    }
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }
    h1, h2, h3, h4, h5 { color: var(--ink-0) !important; font-weight: 800; letter-spacing: -0.5px; }
    p, li, span, label { color: var(--ink-1); }
    [data-testid="stMarkdownContainer"] p { color: var(--ink-1); }
    hr, [data-testid="stDivider"] { border-color: var(--border-soft) !important; }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 0 rgba(244,114,182,0.0); }
        50%      { box-shadow: 0 0 46px rgba(244,114,182,0.42); }
    }
    @keyframes growIn {
        from { opacity: 0; transform: scale(0.55); }
        to   { opacity: 1; transform: scale(1); }
    }
    @keyframes shimmerSweep {
        0%   { left: -150%; }
        55%  { left: 150%; }
        100% { left: 150%; }
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0820 0%, #150c2c 60%, #1a0f2e 100%);
        border-right: 1px solid var(--border-soft);
    }
    section[data-testid="stSidebar"] .stRadio > label {
        color: var(--ink-2) !important;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 12px;
        letter-spacing: 1.2px;
    }
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
        padding: 10px 14px;
        border-radius: var(--radius-sm);
        margin-bottom: 4px;
        transition: background 0.15s ease, transform 0.15s ease;
        color: var(--ink-0) !important;
    }
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {
        background: rgba(244,114,182,0.12);
        transform: translateX(3px);
    }
    .sidebar-brand { text-align: center; padding: 18px 10px 6px 10px; }
    .sidebar-brand .brand-icon {
        font-size: 44px;
        filter: drop-shadow(0 0 18px rgba(244,114,182,0.55));
        animation: float 3.4s ease-in-out infinite;
    }
    .sidebar-brand h2 {
        margin: 6px 0 2px 0;
        font-size: 22px;
        background: linear-gradient(90deg, #c084fc, #f472b6, #fbbf24);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sidebar-brand p { color: var(--ink-2); font-size: 13px; margin: 0; }
    .sidebar-stat {
        display: flex;
        justify-content: space-between;
        font-size: 12.5px;
        color: var(--ink-2);
        padding: 4px 2px;
    }
    .sidebar-stat b { color: var(--ink-0); }
    .hero {
        padding: 44px 42px;
        border-radius: var(--radius-lg);
        background:
            linear-gradient(135deg, rgba(168,85,247,0.30), rgba(244,114,182,0.18), rgba(15,10,28,0.90));
        border: 1px solid var(--border-soft);
        box-shadow: var(--shadow-lg);
        margin-bottom: 26px;
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.6s ease both;
    }
    .hero::before {
        content: "";
        position: absolute;
        top: 0; left: -150%;
        width: 150%; height: 100%;
        background: linear-gradient(120deg, transparent, rgba(255,255,255,0.07), transparent);
        animation: shimmerSweep 7s ease-in-out infinite;
        pointer-events: none;
    }
    .hero::after {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at 90% 0%, rgba(251,191,36,0.14), transparent 45%);
        pointer-events: none;
    }
    .hero-eyebrow {
        display: inline-block;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        color: var(--accent-3);
        background: rgba(251,191,36,0.12);
        border: 1px solid rgba(251,191,36,0.30);
        padding: 5px 12px;
        border-radius: 999px;
        margin-bottom: 14px;
    }
    .hero h1 { font-size: 50px; font-weight: 900; margin: 0; letter-spacing: -2px; line-height: 1.08; }
    .hero p.hero-sub {
        font-size: 17px;
        color: var(--ink-1);
        margin-top: 12px;
        max-width: 640px;
        line-height: 1.6;
    }
    .gradient-text {
        background: linear-gradient(90deg, #c084fc, #f472b6, #fbbf24);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .robot-wrap { display: flex; justify-content: center; align-items: center; height: 210px; }
    .robot {
        width: 145px; height: 125px;
        background: linear-gradient(160deg, #fdf4ff, #f3e1ff);
        border-radius: 35px;
        position: relative;
        box-shadow: 0 0 48px rgba(244,114,182,0.55), inset 0 -6px 12px rgba(168,85,247,0.18);
        animation: float 3s ease-in-out infinite;
    }
    .robot::before { content:""; position:absolute; width:45px; height:15px; background:var(--accent-1); top:-15px; left:50px; border-radius:20px; }
    .robot::after  { content:""; position:absolute; width:8px; height:20px; background:var(--accent-1); top:-30px; left:68px; border-radius:10px; }
    .eye { width:22px; height:22px; background: var(--accent-2); border-radius:50%; position:absolute; top:43px; box-shadow: 0 0 10px rgba(244,114,182,0.65); }
    .eye.left { left: 34px; } .eye.right { right: 34px; }
    .mouth { width:48px; height:12px; border-bottom:4px solid var(--accent-1); border-radius:50%; position:absolute; left:48px; bottom:30px; }
    .ear { width:18px; height:40px; background:#d8b4fe; position:absolute; top:42px; border-radius:12px; }
    .ear.left { left:-13px; } .ear.right { right:-13px; }
    @keyframes float {
        0%   { transform: translateY(0px); }
        50%  { transform: translateY(-12px); }
        100% { transform: translateY(0px); }
    }
    .glass-card {
        padding: 26px;
        border-radius: var(--radius-md);
        background: rgba(20,13,36,0.75);
        border: 1px solid var(--border-soft);
        backdrop-filter: blur(14px);
        box-shadow: var(--shadow-md);
        margin-bottom: 18px;
        height: 100%;
        transition: border 0.2s ease, transform 0.2s ease;
        animation: fadeInUp 0.6s ease both;
    }
    .glass-card:hover { border: 1px solid var(--border-strong); transform: translateY(-4px); }
    .glass-card h2, .glass-card h3 { margin-top: 0; }
    .metric-card {
        padding: 22px;
        border-radius: var(--radius-md);
        background: linear-gradient(135deg, rgba(168,85,247,0.22), rgba(20,13,36,0.88));
        border: 1px solid rgba(244,114,182,0.22);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        animation: fadeInUp 0.6s ease both;
    }
    .metric-card:hover { transform: translateY(-4px); box-shadow: 0 12px 32px rgba(244,114,182,0.25); }
    .metric-icon { font-size: 22px; margin-bottom: 4px; opacity: 0.9; }
    .metric-number {
        font-size: 32px;
        font-weight: 900;
        background: linear-gradient(90deg, #d8b4fe, #f9a8d4, #fde68a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label { color: var(--ink-2); font-size: 13.5px; margin-top: 2px; letter-spacing: 0.3px; }
    .prediction-card {
        padding: 40px;
        border-radius: var(--radius-lg);
        background: linear-gradient(135deg, rgba(251,191,36,0.10), rgba(168,85,247,0.22), rgba(15,10,28,0.95));
        border: 1px solid rgba(244,114,182,0.32);
        box-shadow: var(--shadow-lg);
        text-align: center;
        animation: fadeInUp 0.7s ease both, pulseGlow 3.4s ease-in-out infinite 0.7s;
    }
    .prediction-title {
        color: var(--accent-3);
        font-size: 15px;
        text-transform: uppercase;
        letter-spacing: 2.5px;
        font-weight: 700;
    }
    .prediction-role {
        font-size: 42px;
        font-weight: 900;
        margin-top: 12px;
        background: linear-gradient(90deg, #c084fc, #f472b6, #fbbf24);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    .prediction-footnote { color: var(--ink-2); margin-top: 10px; font-size: 14px; }
    .confidence-ring {
        width: 190px; height: 190px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        margin: 26px auto 8px auto;
        animation: growIn 0.8s cubic-bezier(.34,1.56,.64,1) both, pulseGlow 2.8s ease-in-out infinite 0.8s;
    }
    .confidence-ring-inner {
        width: 152px; height: 152px;
        border-radius: 50%;
        background: radial-gradient(circle, #1a0f2e, #120b24);
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        border: 1px solid rgba(255,255,255,0.10);
    }
    .confidence-pct {
        font-size: 36px;
        font-weight: 900;
        background: linear-gradient(90deg, #f9a8d4, #fde68a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .confidence-caption {
        font-size: 11.5px;
        color: var(--ink-2);
        text-transform: uppercase;
        letter-spacing: 1.6px;
        margin-top: 4px;
        text-align: center;
    }
    .confidence-tag {
        display: inline-block;
        margin-top: 4px;
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 12.5px;
        font-weight: 800;
        letter-spacing: 0.3px;
        background: rgba(52,211,153,0.14);
        border: 1px solid rgba(52,211,153,0.35);
        color: #6ee7b7;
    }
    .course-card {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 16px 18px;
        border-radius: var(--radius-sm);
        background: rgba(30,20,48,0.78);
        border: 1px solid var(--border-soft);
        margin-bottom: 10px;
        transition: transform 0.2s ease, border 0.2s ease, background 0.2s ease;
        animation: fadeInUp 0.5s ease both;
    }
    .course-card:hover {
        transform: translateX(5px);
        border: 1px solid var(--border-strong);
        background: rgba(40,26,62,0.95);
    }
    .course-number {
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
        width: 30px; height: 30px;
        border-radius: 50%;
        font-size: 13px; font-weight: 800;
        background: linear-gradient(135deg, #a855f7, #f472b6);
        color: white;
    }
    .course-name { font-weight: 600; color: var(--ink-0); font-size: 15px; }
    .pill {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.4px;
        background: rgba(251,191,36,0.12);
        border: 1px solid rgba(251,191,36,0.30);
        color: var(--accent-3);
    }
    .stButton > button, .stFormSubmitButton > button {
        width: 100%;
        border-radius: 14px;
        border: none;
        padding: 13px 20px;
        font-weight: 800;
        font-size: 15.5px;
        letter-spacing: 0.3px;
        background: linear-gradient(90deg, #a855f7, #f472b6, #fbbf24);
        background-size: 200% auto;
        color: white;
        transition: transform 0.2s ease, box-shadow 0.2s ease, background-position 0.4s ease;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 34px rgba(244,114,182,0.45);
        background-position: right center;
    }
    div[data-baseweb="select"] > div, .stTextInput > div > div, .stNumberInput > div > div {
        background-color: rgba(20,13,36,0.80) !important;
        border-color: var(--border-soft) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--ink-0) !important;
    }
    .stSlider label, .stSelectbox label, .stRadio label { color: var(--ink-1) !important; font-weight: 600; }
    .section-title { font-size: 28px; font-weight: 850; margin-top: 8px; margin-bottom: 6px; letter-spacing: -0.5px; color: var(--ink-0); }
    .section-subtitle { color: var(--ink-2); margin-bottom: 24px; font-size: 15px; }
    .form-heading {
        font-size: 17px; font-weight: 800; color: var(--ink-0);
        margin: 22px 0 10px 0; padding-bottom: 8px;
        border-bottom: 1px solid var(--border-soft);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        color: var(--ink-2) !important;
        font-weight: 700;
        background: rgba(255,255,255,0.03);
        border-radius: 10px 10px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        border-bottom: 2px solid var(--accent-2) !important;
        background: rgba(244,114,182,0.10);
    }
    div[data-testid="stAlertContentSuccess"],
    div[data-testid="stAlertContentInfo"],
    div[data-testid="stAlertContentWarning"],
    div[data-testid="stAlertContentError"] {
        color: #0a0714 !important;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] { color: var(--ink-0) !important; }
    [data-testid="stMetricLabel"] { color: var(--ink-2) !important; }
    [data-testid="stDataFrame"] * { color: var(--ink-0); }
    .stDataFrame { border-radius: var(--radius-sm); overflow: hidden; }
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: var(--bg-1); }
    ::-webkit-scrollbar-thumb { background: #4c2e6e; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #6b3f96; }
    </style>
    """
)


def hero(title_html, subtitle):
    render_html(
        f"""
        <div class="hero">
        <span class="hero-eyebrow">CareerAI · Internship Project</span>
        <h1>{title_html}</h1>
        <p class="hero-sub">{subtitle}</p>
        </div>
        """
    )


def metric_card(icon, number, label):
    render_html(
        f"""
        <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-number">{number}</div>
        <div class="metric-label">{label}</div>
        </div>
        """
    )


def course_card(index, name):
    render_html(
        f"""
        <div class="course-card">
        <span class="course-number">{index}</span>
        <span class="course-name">{name}</span>
        </div>
        """
    )


def confidence_label(pct):
    if pct >= 85:
        return "Strong Match"
    if pct >= 65:
        return "Good Match"
    if pct >= 45:
        return "Fair Match"
    return "Emerging Match"


def style_results_table(dframe):
    return (
        dframe.style
        .format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1 Score": "{:.2%}"
        })
        .set_properties(**{
            "background-color": "#150f26",
            "color": "#ffffff",
            "border-color": "#2a2440"
        })
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#241a3d"),
                ("color", "#ffffff"),
                ("font-weight", "700")
            ]}
        ])
        .highlight_max(subset=["Accuracy", "F1 Score"], color="#3d2260")
    )


@st.cache_data
def load_data():
    df = pd.read_csv("PS2_Dataset.csv")
    return df


df = load_data()

TARGET = "Suggested Job Role"

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df[TARGET])
classes = target_encoder.classes_
n_classes = len(classes)
X = df.drop(columns=[TARGET])

categorical_columns = X.select_dtypes(include="object").columns.tolist()
numerical_columns = X.select_dtypes(exclude="object").columns.tolist()
feature_order = list(X.columns)

X_catboost = X.copy()
for column in categorical_columns:
    X_catboost[column] = X_catboost[column].fillna("Unknown").astype(str)

ordinal_maps = {}
for column in categorical_columns:
    uniques = sorted(X_catboost[column].unique())
    ordinal_maps[column] = {value: idx for idx, value in enumerate(uniques)}


def ordinal_encode(frame):
    encoded = frame.copy()
    for column in categorical_columns:
        mapping = ordinal_maps[column]
        fallback = len(mapping)
        encoded[column] = (
            encoded[column]
            .astype(str)
            .map(mapping)
            .fillna(fallback)
            .astype(int)
        )
    for column in numerical_columns:
        encoded[column] = pd.to_numeric(encoded[column], errors="coerce").fillna(0)
    return encoded[feature_order]


X_train, X_test, y_train, y_test = train_test_split(
    X_catboost, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train_ord = ordinal_encode(X_train)
X_test_ord = ordinal_encode(X_test)

cat_indices = [X_catboost.columns.get_loc(col) for col in categorical_columns]


def align_proba(model, proba):
    model_classes = np.asarray(model.classes_).ravel().astype(int)
    if np.array_equal(model_classes, np.arange(n_classes)):
        return proba
    aligned = np.zeros((proba.shape[0], n_classes))
    aligned[:, model_classes] = proba
    return aligned


MODEL_CACHE_PATH = "careerai_models.pkl"


@st.cache_resource
def train_models():

    if os.path.exists(MODEL_CACHE_PATH):
        cached = joblib.load(MODEL_CACHE_PATH)
        return (
            cached["trained_models"],
            cached["predictions"],
            cached["results"],
            cached["weight_map"]
        )

    trained_models = {}
    predictions = {}
    test_proba = {}
    scores = []

    cat_model = CatBoostClassifier(
        iterations=900,
        learning_rate=0.035,
        depth=8,
        l2_leaf_reg=4,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
        use_best_model=True,
        early_stopping_rounds=60
    )

    cat_model.fit(
        X_train, y_train,
        cat_features=cat_indices,
        eval_set=(X_test, y_test),
        verbose=False
    )

    cat_proba_test = align_proba(cat_model, cat_model.predict_proba(X_test))
    cat_pred = np.argmax(cat_proba_test, axis=1)

    scores.append({
        "Model": "CatBoost",
        "Accuracy": accuracy_score(y_test, cat_pred),
        "Precision": precision_score(y_test, cat_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, cat_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, cat_pred, average="weighted", zero_division=0)
    })

    trained_models["CatBoost"] = cat_model
    predictions["CatBoost"] = cat_pred
    test_proba["CatBoost"] = cat_proba_test

    extra_model = ExtraTreesClassifier(
        n_estimators=700,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        max_features="sqrt",
        min_samples_leaf=2
    )
    extra_model.fit(X_train_ord, y_train)

    extra_proba_test = align_proba(extra_model, extra_model.predict_proba(X_test_ord))
    extra_pred = np.argmax(extra_proba_test, axis=1)

    scores.append({
        "Model": "Extra Trees",
        "Accuracy": accuracy_score(y_test, extra_pred),
        "Precision": precision_score(y_test, extra_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, extra_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, extra_pred, average="weighted", zero_division=0)
    })

    trained_models["Extra Trees"] = extra_model
    predictions["Extra Trees"] = extra_pred
    test_proba["Extra Trees"] = extra_proba_test

    rf_model = RandomForestClassifier(
        n_estimators=700,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        max_features="sqrt",
        min_samples_leaf=2
    )
    rf_model.fit(X_train_ord, y_train)

    rf_proba_test = align_proba(rf_model, rf_model.predict_proba(X_test_ord))
    rf_pred = np.argmax(rf_proba_test, axis=1)

    scores.append({
        "Model": "Random Forest",
        "Accuracy": accuracy_score(y_test, rf_pred),
        "Precision": precision_score(y_test, rf_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, rf_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, rf_pred, average="weighted", zero_division=0)
    })

    trained_models["Random Forest"] = rf_model
    predictions["Random Forest"] = rf_pred
    test_proba["Random Forest"] = rf_proba_test

    dt_model = DecisionTreeClassifier(
        random_state=42,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced"
    )
    dt_model.fit(X_train_ord, y_train)

    dt_proba_test = align_proba(dt_model, dt_model.predict_proba(X_test_ord))
    dt_pred = np.argmax(dt_proba_test, axis=1)

    scores.append({
        "Model": "Decision Tree",
        "Accuracy": accuracy_score(y_test, dt_pred),
        "Precision": precision_score(y_test, dt_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, dt_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, dt_pred, average="weighted", zero_division=0)
    })

    trained_models["Decision Tree"] = dt_model
    predictions["Decision Tree"] = dt_pred
    test_proba["Decision Tree"] = dt_proba_test

    base_names = ["CatBoost", "Extra Trees", "Random Forest", "Decision Tree"]
    accs = np.array([s["Accuracy"] for s in scores])
    weights = accs ** 2
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(accs)) / len(accs)
    weight_map = dict(zip(base_names, weights))

    ensemble_proba_test = sum(
        weight_map[name] * test_proba[name] for name in base_names
    )
    ensemble_pred = np.argmax(ensemble_proba_test, axis=1)

    scores.append({
        "Model": "CareerAI Ensemble",
        "Accuracy": accuracy_score(y_test, ensemble_pred),
        "Precision": precision_score(y_test, ensemble_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, ensemble_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, ensemble_pred, average="weighted", zero_division=0)
    })

    predictions["CareerAI Ensemble"] = ensemble_pred

    results = pd.DataFrame(scores).sort_values("Accuracy", ascending=False).reset_index(drop=True)

    joblib.dump({
        "trained_models": trained_models,
        "predictions": predictions,
        "results": results,
        "weight_map": weight_map
    }, MODEL_CACHE_PATH)

    return trained_models, predictions, results, weight_map


with st.spinner("🧠 Training CareerAI models..."):
    trained_models, predictions, results, model_weights = train_models()

best_model_name = results.iloc[0]["Model"]
best_accuracy = results.iloc[0]["Accuracy"]


def predict_with_ensemble(user_data_raw):
    user_cat = user_data_raw.copy()
    for col in categorical_columns:
        user_cat[col] = user_cat[col].astype(str)

    user_ord = ordinal_encode(user_data_raw)

    proba_parts = {
        "CatBoost": align_proba(trained_models["CatBoost"], trained_models["CatBoost"].predict_proba(user_cat)),
        "Extra Trees": align_proba(trained_models["Extra Trees"], trained_models["Extra Trees"].predict_proba(user_ord)),
        "Random Forest": align_proba(trained_models["Random Forest"], trained_models["Random Forest"].predict_proba(user_ord)),
        "Decision Tree": align_proba(trained_models["Decision Tree"], trained_models["Decision Tree"].predict_proba(user_ord)),
    }

    ensemble_proba = sum(model_weights[name] * proba_parts[name] for name in proba_parts)
    ensemble_proba = ensemble_proba[0]

    predicted_class = int(np.argmax(ensemble_proba))
    confidence = float(ensemble_proba[predicted_class]) * 100

    return classes[predicted_class], confidence


with st.sidebar:

    render_html(
        """
        <div class="sidebar-brand">
        <div class="brand-icon">🤖</div>
        <h2>CareerAI</h2>
        <p>Intelligent Career Guidance</p>
        </div>
        """
    )

    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠 Dashboard",
            "🔮 Career Predictor",
            "📊 EDA & Analytics",
            "🤖 Model Lab",
            "📚 Course Roadmap",
            "ℹ️ About"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    render_html(
        f"""
        <div class="sidebar-stat"><span>Dataset</span><b>{len(df):,} students</b></div>
        <div class="sidebar-stat"><span>Career classes</span><b>{n_classes}</b></div>
        <div class="sidebar-stat"><span>Engine</span><b>CareerAI Ensemble</b></div>
        <div class="sidebar-stat"><span>Top model</span><b>{best_model_name}</b></div>
        """
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("CareerAI • Internship Project")


if page == "🏠 Dashboard":

    hero(
        '🚀 <span class="gradient-text">CareerAI</span>',
        "Discover the career path that best matches your skills, "
        "interests and personality — then follow a tailored course roadmap to get there."
    )

    col1, col2 = st.columns([1.5, 1])

    with col1:
        render_html(
            """
            <div class="glass-card">
            <h2>🎯 Your Career. Your Future.</h2>
            <p style="font-size:17px; line-height:1.7;">
            CareerAI analyzes your technical skills, logical ability,
            communication, interests and working preferences to
            generate a career-role prediction.
            </p>
            <p style="color:var(--ink-2);">
            After prediction, the system also provides a personalized
            learning roadmap so you know exactly what to study next.
            </p>
            <span class="pill">Weighted 4-model ensemble</span>
            &nbsp;<span class="pill">12 career paths</span>
            </div>
            """
        )

    with col2:
        render_html(
            """
            <div class="robot-wrap">
            <div class="robot">
            <div class="ear left"></div>
            <div class="ear right"></div>
            <div class="eye left"></div>
            <div class="eye right"></div>
            <div class="mouth"></div>
            </div>
            </div>
            """
        )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("🧑‍🎓", f"{len(df):,}", "Student Records")
    with c2:
        metric_card("🧭", n_classes, "Career Roles")
    with c3:
        metric_card("🧩", X.shape[1], "Input Features")
    with c4:
        metric_card("🏆", f"{best_accuracy:.1%}", "Best Test Accuracy")

    render_html('<div class="section-title">🧠 How CareerAI Works</div>')
    render_html(
        '<div class="section-subtitle">From raw student information to a personalized career roadmap.</div>'
    )

    workflow = [
        ("01", "📥", "Load & Preprocess", "Clean and prepare student information."),
        ("02", "🔎", "EDA", "Analyze skills, interests and career distributions."),
        ("03", "🧠", "Train Models", "Train CatBoost, Random Forest, Extra Trees & Decision Tree."),
        ("04", "📈", "Evaluate", "Compare accuracy, precision, recall and F1."),
        ("05", "🎯", "Predict", "Blend all models into one confident recommendation."),
        ("06", "📚", "Recommend", "Generate a career-specific course roadmap.")
    ]

    cols = st.columns(3)

    for i, (number, icon, title, description) in enumerate(workflow):
        with cols[i % 3]:
            render_html(
                f"""
                <div class="glass-card">
                <div style="font-size:32px;">{icon}</div>
                <h3 style="margin-bottom:4px;">{number} · {title}</h3>
                <p style="color:var(--ink-2); margin-bottom:0;">{description}</p>
                </div>
                """
            )


elif page == "🔮 Career Predictor":

    hero(
        "🔮 Career Predictor",
        "Tell CareerAI about yourself. We'll analyze your profile and suggest a career direction."
    )

    with st.form("career_prediction_form"):

        render_html('<div class="form-heading">🧠 Technical &amp; Cognitive Skills</div>')
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            logical = st.slider("Logical Quotient", 0, 10, 5)
        with col2:
            hackathons = st.slider("Hackathons", 0, 10, 2)
        with col3:
            coding = st.slider("Coding Skills", 0, 10, 5)
        with col4:
            speaking = st.slider("Public Speaking", 0, 10, 5)

        render_html('<div class="form-heading">📚 Learning &amp; Development</div>')
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            self_learning = st.selectbox(
                "Self-learning capability?",
                sorted(df["self-learning capability?"].dropna().unique())
            )
        with col2:
            extra_courses = st.selectbox(
                "Extra-courses did",
                sorted(df["Extra-courses did"].dropna().unique())
            )
        with col3:
            certifications = st.selectbox(
                "Certifications",
                sorted(df["certifications"].dropna().unique())
            )
        with col4:
            workshops = st.selectbox(
                "Workshops",
                sorted(df["workshops"].dropna().unique())
            )

        col1, col2 = st.columns(2)

        with col1:
            reading = st.selectbox(
                "Reading & Writing Skills",
                sorted(df["reading and writing skills"].dropna().unique())
            )
        with col2:
            memory = st.selectbox(
                "Memory Capability",
                sorted(df["memory capability score"].dropna().unique())
            )

        render_html('<div class="form-heading">💡 Interests</div>')
        col1, col2 = st.columns(2)

        with col1:
            subjects = st.selectbox(
                "Interested Subjects",
                sorted(df["Interested subjects"].dropna().unique())
            )
        with col2:
            career_area = st.selectbox(
                "Interested Career Area",
                sorted(df["interested career area "].dropna().unique())
            )

        books = st.selectbox(
            "Interested Type of Books",
            sorted(df["Interested Type of Books"].dropna().unique())
        )

        render_html('<div class="form-heading">🧑‍💻 Work Preferences</div>')
        col1, col2 = st.columns(2)

        with col1:
            company = st.selectbox(
                "Type of Company",
                sorted(df["Type of company want to settle in?"].dropna().unique())
            )
        with col2:
            seniors = st.selectbox(
                "Taken Inputs From Seniors/Elders",
                sorted(df["Taken inputs from seniors or elders"].dropna().unique())
            )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            management = st.selectbox(
                "Management or Technical",
                sorted(df["Management or Technical"].dropna().unique())
            )
        with col2:
            hard_worker = st.selectbox(
                "Hard / Smart Worker",
                sorted(df["hard/smart worker"].dropna().unique())
            )
        with col3:
            teamwork = st.selectbox(
                "Worked in Teams?",
                sorted(df["worked in teams ever?"].dropna().unique())
            )
        with col4:
            introvert = st.selectbox(
                "Introvert",
                sorted(df["Introvert"].dropna().unique())
            )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀 PREDICT MY CAREER")

    if submitted:

        user_data = pd.DataFrame({
            "Logical quotient rating": [logical],
            "hackathons": [hackathons],
            "coding skills rating": [coding],
            "public speaking points": [speaking],
            "self-learning capability?": [self_learning],
            "Extra-courses did": [extra_courses],
            "certifications": [certifications],
            "workshops": [workshops],
            "reading and writing skills": [reading],
            "memory capability score": [memory],
            "Interested subjects": [subjects],
            "interested career area ": [career_area],
            "Type of company want to settle in?": [company],
            "Taken inputs from seniors or elders": [seniors],
            "Interested Type of Books": [books],
            "Management or Technical": [management],
            "hard/smart worker": [hard_worker],
            "worked in teams ever?": [teamwork],
            "Introvert": [introvert]
        })[feature_order]

        predicted_role, confidence = predict_with_ensemble(user_data)

        st.markdown("<br>", unsafe_allow_html=True)

        render_html(
            f"""
            <div class="prediction-card">
            <div class="prediction-title">✨ Recommended Career Direction</div>
            <div class="prediction-role">{predicted_role}</div>
            <p class="prediction-footnote">Powered by the CareerAI weighted ensemble
            (CatBoost + Random Forest + Extra Trees + Decision Tree)</p>
            <div class="confidence-ring" style="background: conic-gradient(#f472b6 {confidence:.1f}%, rgba(255,255,255,0.08) {confidence:.1f}% 100%);">
            <div class="confidence-ring-inner">
            <div class="confidence-pct">{confidence:.1f}%</div>
            <div class="confidence-caption">Match Confidence</div>
            </div>
            </div>
            <span class="confidence-tag">{confidence_label(confidence)}</span>
            </div>
            """
        )

        st.markdown("### 📚 Recommended Learning Roadmap")

        course_map = {
            "Applications Developer": [
                "Python / Java Programming", "Object-Oriented Programming",
                "REST API Development", "Application Architecture", "Git & GitHub"
            ],
            "CRM Technical Developer": [
                "CRM Fundamentals", "SQL & Database Management",
                "Python / Java", "API Integration", "Cloud Fundamentals"
            ],
            "Database Developer": [
                "SQL", "PostgreSQL / MySQL", "Database Design",
                "Advanced SQL", "Database Administration"
            ],
            "Mobile Applications Developer": [
                "Android Development", "Kotlin / Java", "Flutter",
                "REST APIs", "Mobile UI/UX"
            ],
            "Network Security Engineer": [
                "Computer Networks", "Cybersecurity Fundamentals",
                "Ethical Hacking", "Network Security", "Security Monitoring"
            ],
            "Software Developer": [
                "Python / Java", "Data Structures & Algorithms",
                "Object-Oriented Programming", "Git & GitHub", "Software Engineering"
            ],
            "Software Engineer": [
                "Data Structures & Algorithms", "System Design",
                "Software Engineering", "Git & GitHub", "Cloud Computing"
            ],
            "Software Quality Assurance (QA) / Testing": [
                "Software Testing Fundamentals", "Manual Testing",
                "Selenium", "API Testing", "Automation Testing"
            ],
            "Systems Security Administrator": [
                "Linux Administration", "Networking", "Cybersecurity",
                "System Hardening", "Security Monitoring"
            ],
            "Technical Support": [
                "Computer Networking", "Operating Systems", "Linux",
                "Troubleshooting", "IT Support Fundamentals"
            ],
            "UX Designer": [
                "UI/UX Fundamentals", "Figma", "User Research",
                "Wireframing", "Design Systems"
            ],
            "Web Developer": [
                "HTML & CSS", "JavaScript", "React",
                "Backend Development", "REST APIs"
            ]
        }

        courses = course_map.get(
            predicted_role,
            ["Python", "SQL", "Git & GitHub", "Data Structures", "Cloud Fundamentals"]
        )

        for i, course in enumerate(courses, start=1):
            course_card(i, course)


elif page == "📊 EDA & Analytics":

    hero("📊 Exploratory Data Analysis", "Understand the dataset before trusting the model.")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("📄", f"{len(df):,}", "Rows")
    with c2:
        metric_card("🧩", X.shape[1], "Features")
    with c3:
        metric_card("🧭", n_classes, "Career Classes")
    with c4:
        metric_card("❓", int(df.isnull().sum().sum()), "Missing Values")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🎯 Target Distribution", "📈 Numerical Features", "🔍 Dataset Preview"])

    with tab1:

        counts = df[TARGET].value_counts().sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(counts.index, counts.values, color=PLOT_C2, edgecolor="none")
        ax.set_xlabel("Number of Students", color=PLOT_FG)
        ax.set_title("Suggested Job Role Distribution", color=PLOT_FG)
        ax.tick_params(colors=PLOT_FG)
        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)

    with tab2:

        numeric_cols = [
            "Logical quotient rating", "hackathons",
            "coding skills rating", "public speaking points"
        ]

        selected = st.selectbox("Choose feature", numeric_cols)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df[selected], bins=15, color=PLOT_C3, edgecolor=PLOT_BG)
        ax.set_title(selected, color=PLOT_FG)
        ax.set_xlabel(selected, color=PLOT_FG)
        ax.set_ylabel("Frequency", color=PLOT_FG)
        ax.tick_params(colors=PLOT_FG)
        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)

    with tab3:

        st.dataframe(df.head(100), use_container_width=True)


elif page == "🤖 Model Lab":

    hero("🤖 Model Laboratory", "Compare the models that power CareerAI's ensemble engine.")

    st.dataframe(style_results_table(results), use_container_width=True)

    st.success(f"🏆 Strongest single component: **{best_model_name}**")

    st.info(
        "CareerAI doesn't rely on a single model — every prediction blends CatBoost, "
        "Random Forest, Extra Trees and a Decision Tree into one weighted ensemble, "
        "which is what the Career Predictor page actually uses."
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    palette = [PLOT_C1, PLOT_C2, PLOT_C3, PLOT_C4, "#c084fc"][:len(results)]
    ax.bar(results["Model"], results["Accuracy"], color=palette)

    ax.axhline(1 / n_classes, linestyle="--", color=PLOT_FG, alpha=0.55, label="Random baseline")
    ax.set_ylim(0, max(0.2, results["Accuracy"].max() * 1.25))
    ax.set_ylabel("Accuracy", color=PLOT_FG)
    ax.set_title("Model Accuracy Comparison", color=PLOT_FG)
    ax.tick_params(colors=PLOT_FG)
    legend = ax.legend(facecolor=PLOT_BG, edgecolor=PLOT_GRID)
    for text in legend.get_texts():
        text.set_color(PLOT_FG)

    plt.xticks(rotation=20)
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)

    best_prediction = predictions[best_model_name]
    cm = confusion_matrix(y_test, best_prediction)

    fig, ax = plt.subplots(figsize=(11, 9))

    sns.heatmap(
        cm, cmap=warm_cmap, annot=False,
        xticklabels=classes, yticklabels=classes,
        ax=ax, cbar_kws={"label": "Count"},
        linewidths=0.4, linecolor=PLOT_BG
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=PLOT_FG)
    cbar.ax.yaxis.label.set_color(PLOT_FG)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=PLOT_FG)

    ax.set_xlabel("Predicted", color=PLOT_FG)
    ax.set_ylabel("Actual", color=PLOT_FG)
    ax.set_title(best_model_name + " Confusion Matrix", color=PLOT_FG)
    ax.tick_params(colors=PLOT_FG)

    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)


elif page == "📚 Course Roadmap":

    hero("📚 Career Course Roadmaps", "Explore the recommended learning path for every career.")

    course_map = {
        "Applications Developer": ["Python / Java", "OOP", "REST APIs", "Application Architecture", "Git & GitHub"],
        "CRM Technical Developer": ["CRM Fundamentals", "SQL", "Programming", "API Integration", "Cloud"],
        "Database Developer": ["SQL", "PostgreSQL", "Database Design", "Advanced SQL", "Database Administration"],
        "Mobile Applications Developer": ["Android", "Kotlin / Java", "Flutter", "REST APIs", "Mobile UI/UX"],
        "Network Security Engineer": ["Networking", "Cybersecurity", "Ethical Hacking", "Network Security", "Security Monitoring"],
        "Software Developer": ["Python / Java", "Data Structures", "Algorithms", "Git", "Software Engineering"],
        "Software Engineer": ["DSA", "System Design", "Software Engineering", "Cloud", "DevOps"],
        "Software Quality Assurance (QA) / Testing": ["Software Testing", "Manual Testing", "Selenium", "API Testing", "Automation"],
        "Systems Security Administrator": ["Linux", "Networking", "Cybersecurity", "System Hardening", "Security Monitoring"],
        "Technical Support": ["Networking", "Operating Systems", "Linux", "Troubleshooting", "IT Support"],
        "UX Designer": ["UI/UX", "Figma", "User Research", "Wireframing", "Design Systems"],
        "Web Developer": ["HTML", "CSS", "JavaScript", "React", "Backend Development"]
    }

    selected_role = st.selectbox("Choose a career", list(course_map.keys()))

    st.markdown(f"### 🚀 Roadmap for {selected_role}")

    for i, course in enumerate(course_map[selected_role], start=1):
        course_card(i, course)


elif page == "ℹ️ About":

    hero("ℹ️ About CareerAI", "An AI-powered career prediction prototype built for the internship project.")

    render_html(
        """
        <div class="glass-card">
        <h2>🎯 Project Objective</h2>
        <p>
        CareerAI attempts to predict a suitable career role from a student's
        technical abilities, interests, learning behaviour and work preferences.
        </p>
        <h2>🧠 Machine Learning</h2>
        <p>
        Four classification algorithms — Decision Tree, Random Forest, Extra Trees
        and CatBoost — are trained and then combined into a single accuracy-weighted
        ensemble, which is the engine that powers every prediction in this app.
        </p>
        <h2>📚 Recommendation Engine</h2>
        <p>
        Once a career role is predicted, a rule-based course recommendation engine
        provides a structured learning roadmap for that career.
        </p>
        <h2>🎨 Interface</h2>
        <p>
        The Streamlit interface provides a dashboard, EDA section, model comparison,
        prediction form, a career-confidence visualization and a course roadmap.
        </p>
        </div>
        """
    )