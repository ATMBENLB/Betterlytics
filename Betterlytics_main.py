from __future__ import annotations

import io
import json
import math
import re
from dataclasses import dataclass
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats as scipy_stats
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="betterlytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

P = {
    "bg":      "#0f172a",
    "bg2":     "#1e293b",
    "bg3":     "#334155",
    "border":  "#334155",
    "text":    "#f1f5f9",
    "muted":   "#94a3b8",
    "blue":    "#0ea5e9",
    "green":   "#10b981",
    "amber":   "#f59e0b",
    "red":     "#ef4444",
    "purple":  "#8b5cf6",
}

CHART_COLORS = [P["blue"], P["green"], P["amber"], P["red"], P["purple"],
                "#06b6d4", "#f97316", "#84cc16", "#ec4899", "#14b8a6"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background: {P['bg']};
    color: {P['text']};
}}
.stApp {{ background: {P['bg']}; }}
.block-container {{ padding: 1.5rem 2rem 3rem; max-width: 1400px; }}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: {P['bg2']};
    border-right: 1px solid {P['border']};
}}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
    color: {P['blue']}; font-family: 'Space Mono', monospace;
    font-size: 0.7rem; letter-spacing: 0.12em; text-transform: uppercase;
}}

/* Hero */
.hero {{ margin-bottom: 1.5rem; }}
.hero-title {{
    font-family: 'Space Mono', monospace; font-size: 2.4rem;
    font-weight: 700; color: {P['text']}; letter-spacing: -0.03em;
    display: inline-block; margin-right: 0.5rem;
}}
.hero-v {{ font-family: 'Space Mono', monospace; font-size: 0.6rem;
    background: {P['blue']}22; color: {P['blue']};
    border: 1px solid {P['blue']}55; border-radius: 4px;
    padding: 2px 6px; vertical-align: middle; }}
.hero-sub {{ color: {P['muted']}; font-size: 0.92rem; margin-top: 0.25rem; }}

/* Metric row */
.mrow {{ display: flex; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 1.25rem; }}
.mc {{
    flex: 1; min-width: 130px;
    background: {P['bg2']}; border: 1px solid {P['border']};
    border-radius: 10px; padding: 0.85rem 1rem; position: relative; overflow: hidden;
}}
.mc::before {{
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, {P['blue']}, {P['green']});
}}
.mc-label {{ font-size: 0.65rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: {P['muted']}; margin-bottom: 0.2rem; }}
.mc-value {{ font-family: 'Space Mono', monospace; font-size: 1.4rem;
    font-weight: 700; color: {P['text']}; line-height: 1.1; }}
.mc-sub {{ font-size: 0.7rem; color: {P['muted']}; margin-top: 0.2rem; }}
.mc-sub.g {{ color: {P['green']}; }}
.mc-sub.a {{ color: {P['amber']}; }}
.mc-sub.r {{ color: {P['red']}; }}

/* Section label */
.slabel {{
    font-family: 'Space Mono', monospace; font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.15em; text-transform: uppercase; color: {P['blue']};
    margin-bottom: 0.6rem; display: flex; align-items: center; gap: 0.5rem;
}}
.slabel::after {{ content: ''; flex: 1; height: 1px; background: {P['border']}; }}

/* Info blocks */
.ib {{ background: {P['blue']}10; border: 1px solid {P['blue']}30;
    border-left: 3px solid {P['blue']}; border-radius: 8px;
    padding: 0.65rem 0.9rem; font-size: 0.84rem; margin-bottom: 0.85rem; }}
.wb {{ background: {P['amber']}10; border: 1px solid {P['amber']}30;
    border-left: 3px solid {P['amber']}; border-radius: 8px;
    padding: 0.65rem 0.9rem; font-size: 0.84rem; margin-bottom: 0.85rem; }}
.gb {{ background: {P['green']}10; border: 1px solid {P['green']}30;
    border-left: 3px solid {P['green']}; border-radius: 8px;
    padding: 0.65rem 0.9rem; font-size: 0.84rem; margin-bottom: 0.85rem; }}
.rb {{ background: {P['red']}10; border: 1px solid {P['red']}30;
    border-left: 3px solid {P['red']}; border-radius: 8px;
    padding: 0.65rem 0.9rem; font-size: 0.84rem; margin-bottom: 0.85rem; }}

/* Insight card */
.icard {{ background: {P['bg2']}; border: 1px solid {P['border']};
    border-radius: 10px; padding: 0.9rem; margin-bottom: 0.65rem; }}
.icard-tag {{ display: inline-block; font-size: 0.6rem; font-weight: 700;
    border-radius: 4px; padding: 2px 6px; text-transform: uppercase;
    letter-spacing: 0.08em; margin-bottom: 0.35rem; }}
.icard-title {{ font-weight: 600; font-size: 0.875rem;
    color: {P['text']}; margin-bottom: 0.25rem; }}
.icard-body {{ font-size: 0.78rem; color: {P['muted']}; line-height: 1.5; }}

/* Tag colors */
.t-blue   {{ background:{P['blue']}22;   color:{P['blue']};   border:1px solid {P['blue']}44; }}
.t-green  {{ background:{P['green']}22;  color:{P['green']};  border:1px solid {P['green']}44; }}
.t-amber  {{ background:{P['amber']}22;  color:{P['amber']};  border:1px solid {P['amber']}44; }}
.t-red    {{ background:{P['red']}22;    color:{P['red']};    border:1px solid {P['red']}44; }}
.t-purple {{ background:{P['purple']}22; color:{P['purple']}; border:1px solid {P['purple']}44; }}

/* Score badge */
.score-badge {{
    display: inline-block; font-family: 'Space Mono', monospace;
    font-size: 1.6rem; font-weight: 700; padding: 0.5rem 1rem;
    border-radius: 10px; margin: 0.5rem 0;
}}
.score-good  {{ background: {P['green']}22; color: {P['green']}; border: 1px solid {P['green']}44; }}
.score-ok    {{ background: {P['amber']}22; color: {P['amber']}; border: 1px solid {P['amber']}44; }}
.score-poor  {{ background: {P['red']}22;   color: {P['red']};   border: 1px solid {P['red']}44; }}

/* Model card */
.model-card {{
    background: {P['bg2']}; border: 1px solid {P['border']};
    border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 0.75rem;
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;
}}
.model-card.best {{ border-color: {P['green']}55; background: {P['green']}08; }}
.model-name {{ font-weight: 600; font-size: 0.9rem; }}
.model-score {{ font-family: 'Space Mono', monospace; font-size: 1.1rem; font-weight: 700; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background: {P['bg2']}; border-radius: 10px; padding: 3px;
    border: 1px solid {P['border']}; gap: 2px;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px; padding: 0.35rem 0.9rem;
    font-size: 0.8rem; font-weight: 500; color: {P['muted']};
    background: transparent;
}}
.stTabs [aria-selected="true"] {{
    background: {P['blue']}22 !important;
    color: {P['blue']} !important; font-weight: 600 !important;
}}
.stTabs [data-baseweb="tab-panel"] {{ padding-top: 1rem; }}

/* Buttons */
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {P['blue']}, #0284c7);
    color: white; border: none; font-family: 'Space Mono', monospace;
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.05em;
    border-radius: 8px; padding: 0.55rem 1.25rem;
}}

/* Expander */
.stExpander {{ border: 1px solid {P['border']} !important;
    border-radius: 10px !important; background: {P['bg2']}; }}

/* Progress */
.stProgress > div > div > div {{
    background: linear-gradient(90deg, {P['blue']}, {P['green']});
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {P['bg2']}; }}
::-webkit-scrollbar-thumb {{ background: {P['border']}; border-radius: 3px; }}

hr {{ border-color: {P['border']}; margin: 1.25rem 0; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# MATPLOTLIB THEME
# ─────────────────────────────────────────────────────────────────

def _set_mpl():
    plt.rcParams.update({
        "figure.facecolor": P["bg2"], "axes.facecolor": P["bg2"],
        "axes.edgecolor": P["border"], "axes.labelcolor": P["muted"],
        "axes.titlecolor": P["text"], "text.color": P["text"],
        "xtick.color": P["muted"], "ytick.color": P["muted"],
        "grid.color": P["border"], "grid.alpha": 0.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "font.family": "monospace",
        "axes.titlesize": 10, "axes.labelsize": 8,
    })

_set_mpl()

def _fig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────
# DATA TYPES
# ─────────────────────────────────────────────────────────────────

@dataclass
class Schema:
    numeric: list[str]
    categorical: list[str]
    text: list[str]
    datetime: list[str]
    boolean: list[str]
    target_candidates: list[str]

# ─────────────────────────────────────────────────────────────────
# LOADING & SCHEMA
# ─────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_csv(raw: bytes) -> pd.DataFrame:
    for kw in [
        {"sep": None, "engine": "python", "encoding": "utf-8"},
        {"sep": ",", "engine": "python", "encoding": "utf-8"},
        {"sep": None, "engine": "python", "encoding": "latin-1"},
        {"sep": "\t", "engine": "python", "encoding": "utf-8"},
    ]:
        try:
            df = pd.read_csv(io.BytesIO(raw), **kw)
            if df.shape[1] >= 1:
                break
        except Exception:
            pass
    df.columns = [_clean_col(c) for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().replace(
                {"": np.nan, "nan": np.nan, "None": np.nan, "null": np.nan, "NA": np.nan})
    return _parse_dates(df)

def _clean_col(n: Any) -> str:
    t = re.sub(r"\s+", "_", str(n).strip())
    t = re.sub(r"[^0-9a-zA-Z_]+", "", t)
    return t or "col"

def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype != object:
            continue
        sample = out[col].dropna().astype(str).head(80)
        if sample.empty:
            continue
        parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
        if parsed.notna().mean() >= 0.8:
            full = pd.to_datetime(out[col], errors="coerce", infer_datetime_format=True)
            if full.notna().mean() >= 0.7:
                out[col] = full
    return out

def infer_schema(df: pd.DataFrame) -> Schema:
    numeric, categorical, text, datetime_cols, boolean = [], [], [], [], []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            boolean.append(col)
        elif pd.api.types.is_datetime64_any_dtype(s):
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            numeric.append(col)
        else:
            nu = s.nunique(dropna=True)
            al = s.dropna().astype(str).str.len().mean() if not s.dropna().empty else 0
            (text if al > 30 or nu > max(25, min(200, int(len(df) * 0.3))) else categorical).append(col)
    targets = []
    for col in numeric:
        if df[col].nunique(dropna=True) >= 5:
            targets.append(col)
    for col in categorical + boolean:
        nu = df[col].nunique(dropna=True)
        if 2 <= nu <= min(20, max(2, int(len(df) * 0.2))):
            targets.append(col)
    return Schema(numeric=numeric, categorical=categorical, text=text,
                  datetime=datetime_cols, boolean=boolean, target_candidates=targets)

# ─────────────────────────────────────────────────────────────────
# PROFILING
# ─────────────────────────────────────────────────────────────────

def basic_profile(df: pd.DataFrame) -> dict:
    cats = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_datetime64_any_dtype(df[c])]
    return {
        "shape": df.shape,
        "duplicates": int(df.duplicated().sum()),
        "mem_mb": float(df.memory_usage(deep=True).sum() / 1e6),
        "missing": df.isna().sum().sort_values(ascending=False),
        "missing_pct": (df.isna().mean() * 100).sort_values(ascending=False),
        "num_summary": df.describe(include=[np.number]).T if df.select_dtypes(include=[np.number]).shape[1] else pd.DataFrame(),
        "cat_summary": pd.DataFrame({
            "unique": df[cats].nunique(),
            "top": [df[c].mode(dropna=True).iloc[0] if not df[c].mode(dropna=True).empty else np.nan for c in cats],
        }) if cats else pd.DataFrame(),
    }

def health_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        mp = round(float(df[col].isna().mean() * 100), 2)
        nu = int(df[col].nunique(dropna=True))
        health = "⚠️ High missing" if mp > 30 else ("🔴 All unique" if nu == len(df) and nu > 100 else "✅ OK")
        rows.append({"Column": col, "Type": str(df[col].dtype), "Missing %": mp,
                     "Unique": nu, "Health": health,
                     "Sample": str(df[col].dropna().iloc[0])[:50] if not df[col].dropna().empty else ""})
    return pd.DataFrame(rows).sort_values("Missing %", ascending=False)

def skewness_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col].dropna()
        if len(s) < 10:
            continue
        sk = float(s.skew())
        rows.append({"Column": col, "Skewness": round(sk, 3),
                     "Kurtosis": round(float(s.kurtosis()), 3),
                     "Shape": "Right-skewed" if sk > 1 else ("Left-skewed" if sk < -1 else ("Normal-like" if abs(sk) < 0.5 else "Mild skew"))})
    return pd.DataFrame(rows).sort_values("Skewness", key=abs, ascending=False)

def outlier_table(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    rows = []
    for col in schema.numeric:
        s = df[col].dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        out_iqr = int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum())
        out_z   = int((np.abs(scipy_stats.zscore(s)) > 3).sum())
        rows.append({"Column": col, "IQR Outliers": out_iqr, "Z>3σ": out_z,
                     "Outlier %": round(out_iqr/len(s)*100, 2),
                     "Min": round(float(s.min()), 4), "Max": round(float(s.max()), 4)})
    return pd.DataFrame(rows).sort_values("Outlier %", ascending=False)

# ─────────────────────────────────────────────────────────────────
# CORRELATIONS & STATS
# ─────────────────────────────────────────────────────────────────

def top_correlations(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return pd.DataFrame(columns=["Feature 1","Feature 2","Correlation","Strength","Relationship"])
    corr = num.corr(numeric_only=True)
    rows = []
    cols = corr.columns.tolist()
    for i, c1 in enumerate(cols):
        for c2 in cols[i+1:]:
            v = corr.loc[c1, c2]
            if pd.notna(v):
                rel = ("Strong positive" if v > .7 else "Moderate positive" if v > .4 else
                       "Strong negative" if v < -.7 else "Moderate negative" if v < -.4 else "Weak")
                rows.append({"Feature 1": c1, "Feature 2": c2,
                             "Correlation": round(float(v), 4),
                             "Strength": round(float(abs(v)), 4),
                             "Relationship": rel})
    return pd.DataFrame(rows).sort_values("Strength", ascending=False).head(n)

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    try:
        ct = pd.crosstab(x, y)
        chi2, _, _, _ = scipy_stats.chi2_contingency(ct)
        n = len(x)
        m = min(ct.shape) - 1
        return float(np.sqrt(chi2 / (n * m))) if m > 0 and n > 0 else 0.0
    except Exception:
        return 0.0

def cat_associations(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    cats = [c for c in schema.categorical if df[c].nunique() <= 30]
    if len(cats) < 2:
        return pd.DataFrame()
    rows = []
    for i, c1 in enumerate(cats):
        for c2 in cats[i+1:]:
            v = cramers_v(df[c1].fillna("_"), df[c2].fillna("_"))
            rows.append({"Feature 1": c1, "Feature 2": c2,
                         "Cramer's V": round(v, 4),
                         "Strength": "Strong" if v > .5 else "Moderate" if v > .3 else "Weak"})
    return pd.DataFrame(rows).sort_values("Cramer's V", ascending=False).head(15)

# ─────────────────────────────────────────────────────────────────
# MODELING
# ─────────────────────────────────────────────────────────────────

def _detect_type(s: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(s):
        return "classification" if s.nunique(dropna=True) <= 10 else "regression"
    return "classification"

def _model_lib(ptype: str) -> dict:
    if ptype == "regression":
        return {
            "Linear Regression":  LinearRegression(),
            "Ridge":              Ridge(alpha=1.0),
            "Decision Tree":      DecisionTreeRegressor(random_state=42, max_depth=6),
            "Random Forest":      RandomForestRegressor(random_state=42, n_estimators=150, max_depth=10),
            "Gradient Boosting":  GradientBoostingRegressor(random_state=42, n_estimators=100),
        }
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree":       DecisionTreeClassifier(random_state=42, max_depth=6),
        "Random Forest":       RandomForestClassifier(random_state=42, n_estimators=150, max_depth=10),
        "Gradient Boosting":   GradientBoostingClassifier(random_state=42, n_estimators=100),
    }

def _auto_models(ptype: str, n_rows: int) -> list[str]:
    if ptype == "regression":
        return ["Linear Regression", "Decision Tree"] if n_rows < 300 else \
               ["Linear Regression", "Random Forest", "Gradient Boosting"]
    return ["Logistic Regression", "Decision Tree"] if n_rows < 300 else \
           ["Logistic Regression", "Random Forest", "Gradient Boosting"]

def _make_prep(df: pd.DataFrame, target: str):
    feats = [c for c in df.columns if c != target]
    X = df[feats].copy()
    dt_cols = [c for c in X.columns if pd.api.types.is_datetime64_any_dtype(X[c])]
    for c in dt_cols:
        X[c+"_yr"] = X[c].dt.year; X[c+"_mo"] = X[c].dt.month
        X[c+"_dw"] = X[c].dt.dayofweek
    X = X.drop(columns=dt_cols, errors="ignore")
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols and X[c].nunique() <= 50]
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
    ], remainder="drop")
    return prep, num_cols, cat_cols

def train_models(df: pd.DataFrame, target: str, model_names: list[str], test_size: float = 0.2) -> dict:
    clean = df.dropna(subset=[target]).copy()
    y = clean[target]
    ptype = _detect_type(y)

    le = None
    if ptype == "classification" and not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)

    prep, num_cols, cat_cols = _make_prep(clean, target)
    X = clean.drop(columns=[target])

    strat = y if ptype == "classification" and y.nunique() > 1 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42, stratify=strat)

    lib = _model_lib(ptype)
    results, details = [], {}

    for name in model_names:
        if name not in lib:
            continue
        pipe = Pipeline([("prep", prep), ("model", lib[name])])
        try:
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_te)
            row = {"Model": name}
            if ptype == "regression":
                row["R²"]   = round(float(r2_score(y_te, preds)), 4)
                row["RMSE"] = round(float(math.sqrt(mean_squared_error(y_te, preds))), 4)
                row["MAE"]  = round(float(mean_absolute_error(y_te, preds)), 4)
                row["_rank"] = row["R²"]
                try:
                    cv = cross_val_score(pipe, X, y, cv=min(3, max(2, len(X)//50)), scoring="r2")
                    row["CV R² (mean)"] = round(float(cv.mean()), 4)
                except Exception:
                    row["CV R² (mean)"] = None
            else:
                row["Accuracy"] = round(float(accuracy_score(y_te, preds)), 4)
                row["_rank"] = row["Accuracy"]
                row["_report"] = classification_report(y_te, preds, output_dict=True, zero_division=0)
                row["_cm"]     = confusion_matrix(y_te, preds)
                try:
                    if y.nunique() == 2:
                        row["AUC-ROC"] = round(float(roc_auc_score(y_te, pipe.predict_proba(X_te)[:,1])), 4)
                except Exception:
                    pass
            results.append(row)
            details[name] = {"pipe": pipe, "y_te": y_te, "preds": preds, "X_te": X_te}
        except Exception as e:
            results.append({"Model": name, "Error": str(e), "_rank": -999})

    lb = pd.DataFrame(results).sort_values("_rank", ascending=False).reset_index(drop=True)
    best = lb.iloc[0]["Model"] if not lb.empty else None

    # Feature importance
    feat_imp = None
    if best and best in details:
        try:
            bp = details[best]["pipe"]
            bm = bp.named_steps["model"]
            pr = bp.named_steps["prep"]
            fnames = []
            for tname, trans, cols in pr.transformers_:
                if tname == "num":
                    fnames.extend(cols)
                elif tname == "cat":
                    ohe = trans.named_steps.get("ohe")
                    if ohe and hasattr(ohe, "get_feature_names_out"):
                        fnames.extend(ohe.get_feature_names_out(cols).tolist())
            if hasattr(bm, "feature_importances_") and fnames:
                imp = bm.feature_importances_
                n = min(len(fnames), len(imp))
                feat_imp = pd.DataFrame({"Feature": fnames[:n], "Importance": imp[:n]}) \
                             .sort_values("Importance", ascending=False).head(15)
            elif hasattr(bm, "coef_") and fnames:
                coef = np.asarray(bm.coef_)
                vals = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
                n = min(len(fnames), len(vals))
                feat_imp = pd.DataFrame({"Feature": fnames[:n], "Importance": vals[:n]}) \
                             .sort_values("Importance", ascending=False).head(15)
        except Exception:
            pass

    return {
        "ptype": ptype, "lb": lb, "details": details,
        "best": best, "target": target,
        "le": le, "feat_imp": feat_imp,
    }

# ─────────────────────────────────────────────────────────────────
# CLUSTERING & ANOMALY
# ─────────────────────────────────────────────────────────────────

def run_clustering(df: pd.DataFrame, max_k: int = 7) -> dict | None:
    num = df.select_dtypes(include=[np.number]).fillna(df.select_dtypes(include=[np.number]).median(numeric_only=True))
    if num.shape[1] < 2 or len(num) < 20:
        return None
    X = StandardScaler().fit_transform(num)
    scores = {}
    for k in range(2, min(max_k, len(num)-1)+1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        if len(set(labels)) >= 2:
            scores[k] = {"score": silhouette_score(X, labels), "labels": labels}
    if not scores:
        return None
    best_k = max(scores, key=lambda k: scores[k]["score"])
    labels = scores[best_k]["labels"]
    coords = PCA(n_components=2, random_state=42).fit_transform(X)
    pca_df = pd.DataFrame({"PC1": coords[:,0], "PC2": coords[:,1],
                           "Group": [f"Cluster {l}" for l in labels]})
    cdf = df.copy()
    cdf["_cluster"] = [f"Cluster {l}" for l in labels]
    profiles = cdf.groupby("_cluster")[num.columns.tolist()].mean().round(2)
    return {"k": best_k, "sil": float(scores[best_k]["score"]),
            "pca_df": pca_df, "clustered": cdf, "profiles": profiles,
            "elbow": {k: scores[k]["score"] for k in scores}}

def run_anomaly(df: pd.DataFrame) -> dict | None:
    num = df.select_dtypes(include=[np.number]).fillna(df.select_dtypes(include=[np.number]).median(numeric_only=True))
    if num.shape[1] < 2 or len(num) < 20:
        return None
    X = StandardScaler().fit_transform(num)
    preds = IsolationForest(contamination=0.05, random_state=42).fit_predict(X)
    scores = IsolationForest(contamination=0.05, random_state=42).fit(X).decision_function(X)
    coords = PCA(n_components=2, random_state=42).fit_transform(X)
    out = df.copy()
    out["_status"] = np.where(preds == -1, "⚠️ Anomaly", "Normal")
    out["_score"]  = np.round(scores, 4)
    pca_df = pd.DataFrame({"PC1": coords[:,0], "PC2": coords[:,1], "Group": out["_status"].values})
    return {"count": int((preds == -1).sum()), "df": out.sort_values("_score"), "pca_df": pca_df}

# ─────────────────────────────────────────────────────────────────
# TEXT ANALYTICS
# ─────────────────────────────────────────────────────────────────

def run_text(df: pd.DataFrame, col: str) -> dict | None:
    series = df[col].dropna().astype(str)
    if len(series) < 5:
        return None
    try:
        cv = CountVectorizer(stop_words="english", ngram_range=(1,2), max_features=30)
        mat = cv.fit_transform(series)
        counts = np.asarray(mat.sum(axis=0)).ravel()
        top = pd.DataFrame({"Term": cv.get_feature_names_out(), "Count": counts}) \
                .sort_values("Count", ascending=False).head(20)
    except Exception:
        top = pd.DataFrame(columns=["Term","Count"])
    try:
        tv = TfidfVectorizer(stop_words="english", max_features=20)
        tm = tv.fit_transform(series)
        tscores = np.asarray(tm.mean(axis=0)).ravel()
        tfidf_df = pd.DataFrame({"Term": tv.get_feature_names_out(), "TF-IDF": np.round(tscores,4)}) \
                     .sort_values("TF-IDF", ascending=False).head(15)
    except Exception:
        tfidf_df = pd.DataFrame(columns=["Term","TF-IDF"])
    pos_w = {"good","great","excellent","love","best","easy","fast","happy","success","awesome","perfect","amazing"}
    neg_w = {"bad","poor","slow","hard","hate","worse","worst","issue","problem","fail","terrible","broken"}
    sents, lens = [], []
    for t in series.head(5000):
        words = re.findall(r"[a-zA-Z']+", t.lower())
        p = sum(1 for w in words if w in pos_w)
        n = sum(1 for w in words if w in neg_w)
        sents.append("Positive" if p > n else "Negative" if n > p else "Neutral")
        lens.append(len(t))
    sc = pd.Series(sents).value_counts().reset_index()
    sc.columns = ["Sentiment","Count"]
    return {"top": top, "tfidf": tfidf_df, "sentiment": sc,
            "avg_len": round(float(np.mean(lens)),1), "n_docs": len(series)}

# ─────────────────────────────────────────────────────────────────
# TIME SERIES
# ─────────────────────────────────────────────────────────────────

def run_ts(df: pd.DataFrame, date_col: str, val_col: str) -> dict | None:
    tmp = df[[date_col, val_col]].dropna().copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna().sort_values(date_col)
    if len(tmp) < 10:
        return None
    ts = tmp.groupby(date_col)[val_col].mean().reset_index()
    w = min(7, max(2, len(ts)//5))
    ts["Trend"] = ts[val_col].rolling(w, min_periods=1).mean()
    ts["Std"]   = ts[val_col].rolling(w, min_periods=1).std().fillna(0)
    latest = float(ts[val_col].iloc[-1])
    ref    = float(ts[val_col].iloc[max(0,len(ts)-4):-1].mean()) if len(ts) > 3 else float(ts[val_col].mean())
    pct    = (latest - ref) / ref * 100 if ref != 0 else 0
    trend  = "📈 Up" if latest > ref * 1.01 else ("📉 Down" if latest < ref * 0.99 else "➡️ Flat")
    monthly = None
    if len(ts) >= 12:
        tmp2 = tmp.copy()
        tmp2["Month"] = tmp2[date_col].dt.month
        m = tmp2.groupby("Month")[val_col].mean().reset_index()
        m["Month"] = m["Month"].map({1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                                      7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"})
        monthly = m
    return {"series": ts, "trend": trend, "latest": latest, "ref": ref,
            "pct": round(pct,2), "monthly": monthly,
            "date_col": date_col, "val_col": val_col}

# ─────────────────────────────────────────────────────────────────
# SCENARIO SIMULATOR
# ─────────────────────────────────────────────────────────────────

def scenario_sim(modeling: dict, base_row: pd.Series, adjustments: dict) -> dict | None:
    if not modeling or not modeling.get("best"):
        return None
    best = modeling["best"]
    det  = modeling["details"].get(best)
    if not det:
        return None
    pipe = det["pipe"]
    orig = base_row.to_dict()
    adj  = {**orig, **adjustments}
    try:
        p_orig = pipe.predict(pd.DataFrame([orig]))[0]
        p_adj  = pipe.predict(pd.DataFrame([adj]))[0]
    except Exception as e:
        return {"error": str(e)}
    delta = None
    try:
        delta = float(p_adj) - float(p_orig)
    except Exception:
        pass
    return {
        "original": p_orig.item() if hasattr(p_orig, "item") else p_orig,
        "adjusted": p_adj.item() if hasattr(p_adj, "item") else p_adj,
        "delta": delta,
    }

# ─────────────────────────────────────────────────────────────────
# TARGET RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────

def target_recs(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    kw = ["target","label","outcome","response","sales","revenue","conversion",
          "profit","price","score","churn","result","status"]
    rows = []
    for col in schema.target_candidates:
        score, why = 0.0, []
        if any(k in col.lower() for k in kw):
            score += 2.5; why.append("name looks outcome-like")
        if df[col].isna().mean() < 0.1:
            score += 1.5; why.append("low missingness")
        if col in schema.numeric and df[col].nunique(dropna=True) >= 10:
            score += 1.5; why.append("good numeric spread")
        elif col not in schema.numeric and 2 <= df[col].nunique(dropna=True) <= 10:
            score += 1.5; why.append("good class count")
        rows.append({"column": col, "score": round(score,2), "why": ", ".join(why)})
    out = pd.DataFrame(rows)
    return out.sort_values("score", ascending=False).reset_index(drop=True) if not out.empty else out

def _health_flags(df: pd.DataFrame, schema: Schema) -> list[str]:
    flags = []
    d = int(df.duplicated().sum())
    if d: flags.append(f"{d:,} duplicate rows")
    if df.isna().mean().mean() > 0.15: flags.append("High overall missingness")
    if len(schema.numeric) < 2: flags.append("Very few numeric columns")
    if len(df) < 50: flags.append("Small sample size")
    return flags

def _auto_mods(df: pd.DataFrame, schema: Schema) -> dict:
    td = target_recs(df, schema)
    dc = schema.datetime[0] if schema.datetime else None
    vc = schema.numeric[0] if schema.numeric else None
    tc = schema.text + [c for c in schema.categorical if df[c].dropna().astype(str).str.len().mean() > 20]
    return {
        "eda": True,
        "corr": len(schema.numeric) >= 2,
        "cat_assoc": len(schema.categorical) >= 2,
        "outliers": len(schema.numeric) >= 1,
        "modeling": not td.empty,
        "clustering": len(schema.numeric) >= 2 and len(df) >= 20,
        "anomaly": len(schema.numeric) >= 2 and len(df) >= 20,
        "text": bool(tc),
        "ts": bool(dc and vc and len(df) >= 10),
    }

# ─────────────────────────────────────────────────────────────────
# FULL SCAN
# ─────────────────────────────────────────────────────────────────

def full_scan(df: pd.DataFrame, schema: Schema, auto_opt: bool,
              target_override: str | None, test_size: float, chosen_models: list[str]) -> dict:
    prog = st.progress(0, text="Initialising scan…")
    log = []
    scan: dict[str, Any] = {"profile": basic_profile(df), "flags": _health_flags(df, schema)}
    mods = _auto_mods(df, schema)
    td = target_recs(df, schema)
    target = target_override or (td.iloc[0]["column"] if not td.empty else None)
    scan["target_recs"] = td

    prog.progress(8, text="Profiling data quality…")
    scan["skewness"] = skewness_table(df)
    scan["outliers"] = outlier_table(df, schema)
    scan["corr"] = top_correlations(df)
    scan["cat_assoc"] = cat_associations(df, schema) if mods["cat_assoc"] else pd.DataFrame()
    log.append({"Step": "Profile + Correlations", "Status": "✅"})
    prog.progress(22, text="Computing correlations…")

    if mods["modeling"] and target:
        prog.progress(40, text=f"Training models → '{target}'…")
        model_list = chosen_models if chosen_models else (
            _auto_models(_detect_type(df[target]), len(df)) if auto_opt else list(_model_lib(_detect_type(df[target])).keys())
        )
        try:
            scan["modeling"] = train_models(df, target, model_list, test_size)
            log.append({"Step": "Modeling", "Status": "✅", "Target": target, "Models": ", ".join(model_list)})
        except Exception as e:
            scan["modeling"] = None
            log.append({"Step": "Modeling", "Status": f"❌ {e}"})
    else:
        scan["modeling"] = None
        log.append({"Step": "Modeling", "Status": "⏭️ No target found"})

    if mods["clustering"]:
        prog.progress(58, text="Finding natural segments…")
        scan["clustering"] = run_clustering(df)
        log.append({"Step": "Clustering", "Status": "✅" if scan["clustering"] else "⏭️"})
    else:
        scan["clustering"] = None

    if mods["anomaly"]:
        prog.progress(70, text="Detecting anomalies…")
        scan["anomaly"] = run_anomaly(df)
        log.append({"Step": "Anomaly Detection", "Status": "✅" if scan["anomaly"] else "⏭️"})
    else:
        scan["anomaly"] = None

    tc = (schema.text + [c for c in schema.categorical
                         if df[c].dropna().astype(str).str.len().mean() > 20])
    text_col = tc[0] if tc else None
    if mods["text"] and text_col:
        prog.progress(82, text=f"Text analytics on '{text_col}'…")
        scan["text"] = run_text(df, text_col)
        scan["text_col"] = text_col
        log.append({"Step": "Text Analytics", "Status": "✅"})
    else:
        scan["text"] = None; scan["text_col"] = None

    dc = schema.datetime[0] if schema.datetime else None
    vc = schema.numeric[0] if schema.numeric else None
    if mods["ts"] and dc and vc:
        prog.progress(92, text="Time series analysis…")
        scan["ts"] = run_ts(df, dc, vc)
        scan["date_col"] = dc; scan["val_col"] = vc
        log.append({"Step": "Time Series", "Status": "✅"})
    else:
        scan["ts"] = None

    prog.progress(100, text="Done!")
    scan["log"] = log
    return scan

# ─────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────

def chart_corr_heatmap(df: pd.DataFrame):
    num = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(9,7))
    if num.shape[1] < 2:
        ax.text(.5,.5,"Need ≥ 2 numeric columns",ha="center",va="center",color=P["muted"]); ax.axis("off"); return fig
    corr = num.corr(numeric_only=True)
    n = len(corr)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(corr.index, fontsize=7)
    for i in range(n):
        for j in range(n):
            v = corr.iloc[i,j]
            ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=6,
                    color="white" if abs(v)>.6 else P["muted"])
    fig.colorbar(im,ax=ax,shrink=.8); ax.set_title("Correlation Matrix"); fig.tight_layout(); return fig

def chart_missing(df: pd.DataFrame):
    miss = (df.isna().mean()*100).sort_values(ascending=False).head(20)
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(9,4))
    if miss.empty:
        ax.text(.5,.5,"✅ No missing values",ha="center",va="center",color=P["green"],fontsize=13); ax.axis("off"); return fig
    colors = [P["red"] if v>30 else (P["amber"] if v>10 else P["blue"]) for v in miss.values]
    bars = ax.barh(miss.index, miss.values, color=colors, alpha=.85, height=.6)
    for bar, val in zip(bars, miss.values):
        ax.text(val+.3, bar.get_y()+bar.get_height()/2, f"{val:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("Missing %"); ax.set_title("Missing Values by Column"); ax.set_xlim(0, miss.values.max()*1.2)
    fig.tight_layout(); return fig

def chart_distributions(df: pd.DataFrame, cols: list[str]):
    n = len(cols)
    if not n: return None
    ncols = min(3,n); nrows = math.ceil(n/ncols)
    fig, axes = plt.subplots(nrows,ncols,figsize=(ncols*4, nrows*3))
    axes = np.array(axes).flatten() if n>1 else [axes]
    for i,col in enumerate(cols):
        ax = axes[i]; data = df[col].dropna()
        ax.hist(data, bins=min(30,max(5,len(data)//10)), color=CHART_COLORS[i%len(CHART_COLORS)], alpha=.75, edgecolor=P["border"])
        ax.axvline(data.mean(), color=P["amber"], linewidth=1.5, linestyle="--", label=f"μ={data.mean():.2f}")
        ax.set_title(col[:22], fontsize=9); ax.legend(fontsize=7)
    for ax in axes[n:]: ax.axis("off")
    fig.suptitle("Distributions", fontsize=11); fig.tight_layout(); return fig

def chart_scatter(df: pd.DataFrame, x: str, y: str, color_col: str | None = None):
    fig, ax = plt.subplots(figsize=(8,5))
    if color_col and df[color_col].nunique() <= 10:
        for i,(g,sub) in enumerate(df.groupby(color_col)):
            ax.scatter(sub[x],sub[y],label=str(g)[:20],color=CHART_COLORS[i%len(CHART_COLORS)],alpha=.6,s=25,edgecolors="none")
        ax.legend(fontsize=8,title=color_col)
    else:
        ax.scatter(df[x],df[y],color=P["blue"],alpha=.5,s=20,edgecolors="none")
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(f"{x} vs {y}"); fig.tight_layout(); return fig

def chart_pca(pca_df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(8,6))
    for i,(lbl,sub) in enumerate(pca_df.groupby("Group")):
        ax.scatter(sub["PC1"],sub["PC2"],label=str(lbl),color=CHART_COLORS[i%len(CHART_COLORS)],alpha=.7,s=30,edgecolors="none")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title(title); ax.legend(fontsize=8); fig.tight_layout(); return fig

def chart_silhouette(elbow: dict):
    fig, ax = plt.subplots(figsize=(7,4))
    ks = sorted(elbow); best = max(elbow, key=elbow.get)
    ax.plot(ks,[elbow[k] for k in ks],marker="o",color=P["blue"],linewidth=2)
    ax.scatter([best],[elbow[best]],color=P["green"],s=120,zorder=5,label=f"Best k={best}")
    ax.set_xlabel("k"); ax.set_ylabel("Silhouette"); ax.set_title("Silhouette by k"); ax.legend(); fig.tight_layout(); return fig

def chart_feat_imp(feat_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8,5))
    top = feat_df.head(12).sort_values("Importance")
    ax.barh(top["Feature"],top["Importance"],color=CHART_COLORS[:len(top)],alpha=.85,height=.6)
    ax.set_xlabel("Importance"); ax.set_title("Feature Importance"); fig.tight_layout(); return fig

def chart_confusion(cm: np.ndarray):
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, cmap="Blues")
    n = cm.shape[0]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    for i in range(n):
        for j in range(n):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center",
                    color="white" if cm[i,j]>cm.max()/2 else P["text"],fontsize=10)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
    fig.colorbar(im,ax=ax,shrink=.8); fig.tight_layout(); return fig

def chart_actual_vs_pred(y_te, preds):
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(y_te, preds, alpha=.5, color=P["blue"], s=20, edgecolors="none")
    lims = [min(y_te.min(),preds.min()), max(y_te.max(),preds.max())]
    ax.plot(lims,lims,color=P["amber"],linewidth=1.5,linestyle="--",label="Perfect")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title("Actual vs Predicted"); ax.legend()
    fig.tight_layout(); return fig

def chart_residuals(y_te, preds):
    resid = np.array(preds) - np.array(y_te)
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    axes[0].scatter(preds, resid, alpha=.5, color=P["blue"], s=20, edgecolors="none")
    axes[0].axhline(0, color=P["amber"], linewidth=1.5, linestyle="--")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual"); axes[0].set_title("Residual Plot")
    axes[1].hist(resid, bins=30, color=P["blue"], alpha=.75, edgecolor=P["border"])
    axes[1].set_xlabel("Residual"); axes[1].set_title("Residual Distribution")
    fig.tight_layout(); return fig

def chart_cat_bar(df: pd.DataFrame, col: str, n: int = 15):
    counts = df[col].value_counts().head(n)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(range(len(counts)),counts.values,color=CHART_COLORS[:len(counts)],alpha=.85)
    ax.set_xticks(range(len(counts))); ax.set_xticklabels(counts.index,rotation=45,ha="right",fontsize=8)
    ax.set_title(f"Top Values — {col}"); ax.set_ylabel("Count"); fig.tight_layout(); return fig

def chart_text_terms(top: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8,5))
    data = top.head(15).sort_values("Count")
    ax.barh(data["Term"],data["Count"],color=CHART_COLORS[:len(data)],alpha=.85,height=.6)
    ax.set_xlabel("Frequency"); ax.set_title("Top Terms"); fig.tight_layout(); return fig

def chart_sentiment(sc: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5,5))
    cm = {"Positive":P["green"],"Negative":P["red"],"Neutral":P["muted"]}
    colors = [cm.get(s,P["blue"]) for s in sc["Sentiment"]]
    wedges,texts,autos = ax.pie(sc["Count"],labels=sc["Sentiment"],colors=colors,
                                 autopct="%1.1f%%",startangle=90,
                                 wedgeprops={"edgecolor":P["bg"],"linewidth":2})
    for t in texts: t.set_color(P["text"]); t.set_fontsize(9)
    for a in autos: a.set_color(P["bg"]); a.set_fontsize(8); a.set_fontweight("bold")
    ax.set_title("Sentiment"); fig.tight_layout(); return fig

def chart_ts(ts_result: dict):
    ts = ts_result["series"]
    dc = ts_result["date_col"]; vc = ts_result["val_col"]
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(ts[dc],ts[vc],color=P["blue"],alpha=.6,linewidth=1.2,label="Actual")
    ax.plot(ts[dc],ts["Trend"],color=P["green"],linewidth=2.5,label="Trend")
    ax.fill_between(ts[dc],ts["Trend"]-ts["Std"],ts["Trend"]+ts["Std"],alpha=.12,color=P["green"])
    ax.set_xlabel(dc); ax.set_ylabel(vc); ax.set_title(f"Time Trend — {vc}"); ax.legend(fontsize=8)
    fig.autofmt_xdate(); fig.tight_layout(); return fig

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def mc(label,value,sub="",sub_cls=""):
    sc = f'<div class="mc-sub {sub_cls}">{sub}</div>' if sub else ""
    return f'<div class="mc"><div class="mc-label">{label}</div><div class="mc-value">{value}</div>{sc}</div>'

def slabel(t): st.markdown(f'<div class="slabel">{t}</div>', unsafe_allow_html=True)

def icard(tag,tag_cls,title,body):
    st.markdown(f'<div class="icard"><span class="icard-tag {tag_cls}">{tag}</span>'
                f'<div class="icard-title">{title}</div><div class="icard-body">{body}</div></div>',
                unsafe_allow_html=True)

def ib(msg): st.markdown(f'<div class="ib">{msg}</div>', unsafe_allow_html=True)
def wb(msg): st.markdown(f'<div class="wb">{msg}</div>', unsafe_allow_html=True)
def gb(msg): st.markdown(f'<div class="gb">{msg}</div>', unsafe_allow_html=True)
def rb(msg): st.markdown(f'<div class="rb">{msg}</div>', unsafe_allow_html=True)

def score_badge(val, ptype):
    if ptype == "regression":
        cls = "score-good" if val>=.6 else ("score-ok" if val>=.3 else "score-poor")
        return f'<div class="score-badge {cls}">R² {val:.4f}</div>'
    cls = "score-good" if val>=.8 else ("score-ok" if val>=.6 else "score-poor")
    return f'<div class="score-badge {cls}">{val:.1%} acc</div>'

def safe_json(scan: dict) -> str:
    def cvt(o):
        if isinstance(o, pd.DataFrame): return o.to_dict(orient="records")
        if isinstance(o, pd.Series): return o.to_dict()
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.integer, np.floating)): return o.item()
        return str(o)
    return json.dumps(scan, default=cvt, indent=2)

# ─────────────────────────────────────────────────────────────────
# ██  APP START  ██
# ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <span class="hero-title">betterlytics</span>
  <span class="hero-v">v5</span>
  <div class="hero-sub">Drop a CSV → get correlations, predictions, segments, anomalies, text analytics, time trends & what-if simulations — all in one scan.</div>
</div>""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────
with st.sidebar:
    st.markdown("### ① Upload")
    uploaded = st.file_uploader("Choose a CSV", type=["csv"])

    st.markdown("---")
    st.markdown("### ② Settings")
    auto_opt = st.checkbox("Auto-optimize models", value=True)
    test_size = st.slider("Test holdout size", 0.1, 0.4, 0.2, 0.05)

    st.markdown("---")
    st.markdown("### ③ View options")
    show_raw    = st.checkbox("Show raw data preview", value=False)
    show_health = st.checkbox("Show column health table", value=True)

    st.markdown("---")
    st.markdown("""<div style="font-size:0.7rem;color:#64748b;line-height:1.7">
    Works best with:<br>
    • CSVs with ≥2 numeric cols<br>
    • 50+ rows for modeling<br>
    • A datetime col for trends<br>
    • A text col for NLP
    </div>""", unsafe_allow_html=True)

if uploaded is None:
    st.markdown("""<div style="text-align:center;padding:5rem 2rem">
        <div style="font-size:3.5rem;margin-bottom:1rem">📁</div>
        <div style="font-size:1.2rem;font-weight:600;margin-bottom:0.4rem">Upload a CSV to begin</div>
        <div style="color:#64748b;font-size:0.88rem">betterlytics will auto-detect structure and surface the strongest patterns</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── LOAD ─────────────────────────────────────
with st.spinner("Loading…"):
    try:
        df = load_csv(uploaded.getvalue())
    except Exception as e:
        st.error(f"Could not read file: {e}"); st.stop()

schema = infer_schema(df)
profile = basic_profile(df)
h_df    = health_report(df)
mods    = _auto_mods(df, schema)
td      = target_recs(df, schema)
rec_t   = td.iloc[0]["column"] if not td.empty else None
tc_list = schema.text + [c for c in schema.categorical if df[c].dropna().astype(str).str.len().mean() > 20]
rec_txt = tc_list[0] if tc_list else None
rec_d   = schema.datetime[0] if schema.datetime else None
rec_v   = schema.numeric[0] if schema.numeric else None
flags   = _health_flags(df, schema)

# ── TOP METRICS ───────────────────────────────
miss_avg = float(df.isna().mean().mean()*100)
mc_html = (
    mc("Rows", f"{profile['shape'][0]:,}") +
    mc("Columns", str(profile["shape"][1])) +
    mc("Numeric cols", str(len(schema.numeric))) +
    mc("Duplicates", str(profile["duplicates"]),
       "⚠️ Remove before modeling" if profile["duplicates"] else "✅ Clean",
       "a" if profile["duplicates"] else "g") +
    mc("Avg missing", f"{miss_avg:.1f}%",
       "⚠️ Imputation advised" if miss_avg>10 else "✅ Looks good",
       "a" if miss_avg>10 else "g") +
    mc("File size", f"{profile['mem_mb']:.2f} MB")
)
st.markdown(f'<div class="mrow">{mc_html}</div>', unsafe_allow_html=True)

if flags:
    wb("⚠️ <strong>Flags:</strong> " + " &nbsp;·&nbsp; ".join(flags))

if show_raw:
    with st.expander("Raw data preview", expanded=False):
        st.dataframe(df.head(200), use_container_width=True)

if show_health:
    with st.expander("Column health table", expanded=False):
        st.dataframe(h_df, use_container_width=True, hide_index=True)

with st.expander("Schema & auto-recommendations", expanded=False):
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**Column types**")
        st.json({"numeric":schema.numeric,"categorical":schema.categorical,
                 "text":schema.text,"datetime":schema.datetime,"boolean":schema.boolean}, expanded=False)
    with c2:
        st.markdown("**Smart picks**")
        st.write(f"🎯 Target: **{rec_t or 'none found'}**")
        st.write(f"📝 Text col: **{rec_txt or 'none found'}**")
        st.write(f"📅 Date / Value: **{rec_d or 'none'} / {rec_v or 'none'}**")

# ── SCAN CONTROLS ─────────────────────────────
st.markdown("---")
slabel("CONFIGURE & LAUNCH SCAN")

col_t, col_m, col_btn = st.columns([2, 3, 1])
with col_t:
    sel_target = st.selectbox("Target column",
        options=[None]+schema.target_candidates,
        index=1 if rec_t else 0,
        format_func=lambda x: f"{x} (recommended)" if x==rec_t else (x or "— none —"))
with col_m:
    all_models_reg = list(_model_lib("regression").keys())
    all_models_cls = list(_model_lib("classification").keys())
    ptype_hint = "regression" if (sel_target and sel_target in schema.numeric and df[sel_target].nunique()>10) else "classification"
    all_m = all_models_reg if ptype_hint=="regression" else all_models_cls
    default_m = _auto_models(ptype_hint, len(df))
    sel_models = st.multiselect("Models to train", all_m, default=default_m,
                                help="Choose which models to compare in the leaderboard")
with col_btn:
    st.markdown("<div style='margin-top:1.85rem'></div>", unsafe_allow_html=True)
    run_btn = st.button("⚡ Run Scan", type="primary", use_container_width=True)

if run_btn:
    with st.spinner(""):
        scan = full_scan(df, schema, auto_opt, sel_target, test_size, sel_models)
    st.session_state["scan"] = scan
    gb("✅ Scan complete!")

if "scan" not in st.session_state:
    ib("👆 Click <strong>⚡ Run Scan</strong> to generate the full dashboard.")
    st.stop()

scan = st.session_state["scan"]
modeling   = scan.get("modeling")
clustering = scan.get("clustering")
anomaly    = scan.get("anomaly")
corr_df    = scan.get("corr")

# ── EXEC SNAPSHOT ────────────────────────────
st.markdown("---")
slabel("EXECUTIVE SNAPSHOT")
best_score_str = "—"
best_name = "—"
if modeling and not modeling["lb"].empty:
    lead = modeling["lb"].iloc[0]
    best_name = lead["Model"]
    if modeling["ptype"] == "classification":
        best_score_str = f"{lead.get('Accuracy',0):.1%}"
    else:
        best_score_str = f"R² {lead.get('R²',lead.get('R2',0)):.3f}"

snap = (
    mc("Best target", modeling["target"] if modeling else (rec_t or "None")) +
    mc("Best model score", best_score_str, best_name) +
    mc("Segments found", str(clustering["k"]) if clustering else "—",
       f"Sil {clustering['sil']:.2f}" if clustering else "") +
    mc("Anomalies", str(anomaly["count"]) if anomaly else "—",
       "⚠️ Review flagged rows" if anomaly and anomaly["count"]>0 else "")
)
st.markdown(f'<div class="mrow">{snap}</div>', unsafe_allow_html=True)

# ── INSIGHT CARDS ─────────────────────────────
slabel("TOP INSIGHTS")
if corr_df is not None and not corr_df.empty:
    t = corr_df.iloc[0]
    icard("Correlation","t-blue",
          f"{t['Feature 1']} ↔ {t['Feature 2']}",
          f"r = {t['Correlation']:.3f} — {t['Relationship']}. Strongest linear relationship in the dataset.")
if modeling and not modeling["lb"].empty:
    lead = modeling["lb"].iloc[0]
    icard("Prediction","t-green",
          f"Best model: {lead['Model']}",
          f"Score: {best_score_str}. {'Strong enough for pilot experiments.' if lead.get('Accuracy',lead.get('R²',lead.get('R2',0)))>0.6 else 'More data or features may help.'}")
if anomaly and anomaly["count"]>0:
    icard("Anomaly","t-red",
          f"{anomaly['count']} anomalous records detected",
          "Unusual combinations of values — may be data errors, rare events, or high-value exceptions.")
if clustering:
    icard("Segmentation","t-amber",
          f"{clustering['k']} natural clusters found",
          f"Silhouette {clustering['sil']:.3f}. {'Strong separation.' if clustering['sil']>.4 else 'Moderate — treat as directional.'}")

# ══════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════

(t_overview, t_corr, t_model, t_seg, t_text,
 t_ts, t_eda, t_sim, t_log) = st.tabs([
    "📊 Overview", "🔗 Correlations", "🤖 Modeling",
    "🔍 Segments", "📝 Text", "📈 Time Trends",
    "🛠️ EDA Tools", "🧪 Scenario Lab", "📋 Run Log",
])

# ── OVERVIEW ──────────────────────────────────
with t_overview:
    slabel("DATA QUALITY")
    c1,c2 = st.columns(2)
    with c1: _fig(chart_missing(df))
    with c2:
        sk = scan.get("skewness", pd.DataFrame())
        if not sk.empty: st.dataframe(sk, use_container_width=True, hide_index=True)

    slabel("NUMERIC SUMMARY")
    if not profile["num_summary"].empty:
        st.dataframe(profile["num_summary"].style.format(precision=3), use_container_width=True)

    slabel("OUTLIER REPORT")
    out_df = scan.get("outliers", pd.DataFrame())
    if not out_df.empty: st.dataframe(out_df, use_container_width=True, hide_index=True)
    else: st.info("No numeric columns to check.")

# ── CORRELATIONS ──────────────────────────────
with t_corr:
    slabel("NUMERIC CORRELATIONS")
    c1,c2 = st.columns([1.3,1])
    with c1: _fig(chart_corr_heatmap(df))
    with c2:
        if corr_df is not None and not corr_df.empty:
            st.dataframe(corr_df, use_container_width=True, hide_index=True)
        else: st.info("Need ≥ 2 numeric columns.")

    ca = scan.get("cat_assoc", pd.DataFrame())
    if not ca.empty:
        slabel("CATEGORICAL ASSOCIATIONS (CRAMER'S V)")
        ib("Cramer's V measures association between categorical columns — 0 = none, 1 = perfect.")
        st.dataframe(ca, use_container_width=True, hide_index=True)

    if len(schema.numeric) >= 2:
        slabel("SCATTER EXPLORER")
        sc1,sc2,sc3 = st.columns(3)
        xc = sc1.selectbox("X axis", schema.numeric, key="sx")
        yc = sc2.selectbox("Y axis", schema.numeric, index=min(1,len(schema.numeric)-1), key="sy")
        cc = sc3.selectbox("Color by", [None]+schema.categorical+schema.boolean,
                           format_func=lambda x: x or "— none —", key="sc")
        if xc != yc: _fig(chart_scatter(df,xc,yc,cc))

# ── MODELING ──────────────────────────────────
with t_model:
    if not modeling:
        wb("No model ran. Select a target column and click ⚡ Run Scan.")
    else:
        lb = modeling["lb"]
        ptype = modeling["ptype"]
        best = modeling["best"]
        details = modeling["details"]

        slabel("MODEL LEADERBOARD")

        # Render model cards
        display_skip = {"_rank","_report","_cm","Error"}
        for i, row in lb.iterrows():
            is_best = row["Model"] == best
            card_cls = "model-card best" if is_best else "model-card"
            score_val = row.get("Accuracy", row.get("R²", row.get("R2", None)))
            score_html = ""
            if score_val is not None and not (isinstance(score_val, float) and math.isnan(score_val)):
                score_cls = ("score-good" if (score_val>.8 if ptype=="classification" else score_val>.6)
                             else ("score-ok" if (score_val>.6 if ptype=="classification" else score_val>.3)
                             else "score-poor"))
                score_fmt = f"{score_val:.1%}" if ptype=="classification" else f"R² {score_val:.4f}"
                score_html = f'<span class="score-badge {score_cls}" style="font-size:.9rem;padding:.25rem .6rem">{score_fmt}</span>'

            extras = ""
            for k,v in row.items():
                if k in display_skip or k == "Model" or (isinstance(v,float) and math.isnan(v)): continue
                if k in ("Accuracy","R²","R2"): continue
                extras += f'<span style="font-size:.75rem;color:{P["muted"]};margin-left:1rem">{k}: <strong style="color:{P["text"]}">{v}</strong></span>'

            crown = "👑 " if is_best else ""
            st.markdown(f'''
            <div class="{card_cls}">
                <div>
                    <div class="model-name">{crown}{row["Model"]}</div>
                    <div style="font-size:.72rem;color:{P["muted"]}">{extras}</div>
                </div>
                {score_html}
            </div>''', unsafe_allow_html=True)

        if best and best in details:
            det = details[best]
            y_te, preds = det["y_te"], det["preds"]

            st.markdown("---")
            slabel(f"BEST MODEL DETAILS — {best}")

            if ptype == "regression":
                c1,c2 = st.columns(2)
                with c1:
                    _fig(chart_actual_vs_pred(y_te, preds))
                with c2:
                    _fig(chart_residuals(y_te, preds))

                # Regression metrics explained
                r2_val = float(r2_score(y_te, preds))
                rmse_v = float(math.sqrt(mean_squared_error(y_te, preds)))
                mae_v  = float(mean_absolute_error(y_te, preds))
                m1,m2,m3 = st.columns(3)
                with m1:
                    st.markdown(f"""**R² Score**
{score_badge(r2_val,'regression')}""", unsafe_allow_html=True)
                    ib(f"R² of <strong>{r2_val:.4f}</strong> means the model explains <strong>{r2_val*100:.1f}%</strong> of variance in the target. Higher is better (max 1.0).")
                with m2:
                    st.metric("RMSE", f"{rmse_v:.4f}")
                    ib(f"Root Mean Squared Error: on average, predictions are off by ~<strong>{rmse_v:.4f}</strong> units.")
                with m3:
                    st.metric("MAE", f"{mae_v:.4f}")
                    ib(f"Mean Absolute Error: average absolute deviation from true values.")

            else:
                acc = float(accuracy_score(y_te, preds))
                st.markdown(f"""**Accuracy: {acc:.1%}**
{score_badge(acc,'classification')}""", unsafe_allow_html=True)
                ib(f"Accuracy of <strong>{acc:.1%}</strong> means the model correctly predicted <strong>{acc*100:.1f}%</strong> of test cases. "
                   f"Consider precision/recall for imbalanced classes.")

                c1,c2 = st.columns(2)
                with c1:
                    slabel("CONFUSION MATRIX")
                    row_data = lb.set_index("Model").loc[best]
                    cm_data = row_data.get("_cm")
                    if cm_data is not None and not isinstance(cm_data, float):
                        _fig(chart_confusion(cm_data))
                with c2:
                    slabel("CLASSIFICATION REPORT")
                    rep = row_data.get("_report")
                    if isinstance(rep, dict):
                        rep_df = pd.DataFrame(rep).transpose()
                        st.dataframe(rep_df.style.format(precision=3), use_container_width=True)

            # Feature importance
            if modeling.get("feat_imp") is not None:
                st.markdown("---")
                slabel("FEATURE IMPORTANCE")
                c1,c2 = st.columns([1.4,1])
                with c1: _fig(chart_feat_imp(modeling["feat_imp"]))
                with c2:
                    st.dataframe(modeling["feat_imp"], use_container_width=True, hide_index=True)
                    ib("Features with higher importance had more influence on the model's predictions. "
                       "These are strong candidates for deeper investigation.")

        # Change target + rerun inline
        st.markdown("---")
        slabel("RE-RUN WITH DIFFERENT SETTINGS")
        ib("Change the target or model selection above and click ⚡ Run Scan again to compare results.")

# ── SEGMENTATION ──────────────────────────────
with t_seg:
    c1,c2 = st.columns(2)
    with c1:
        slabel("CLUSTERING")
        if clustering:
            km1,km2 = st.columns(2)
            km1.metric("Best k", clustering["k"])
            km2.metric("Silhouette", f"{clustering['sil']:.3f}")
            _fig(chart_pca(clustering["pca_df"], "Cluster Map (PCA 2D)"))
            _fig(chart_silhouette(clustering["elbow"]))
            slabel("CLUSTER PROFILES")
            st.dataframe(clustering["profiles"], use_container_width=True)
            st.download_button("⬇️ Download clustered data",
                               data=clustering["clustered"].to_csv(index=False).encode(),
                               file_name="betterlytics_clusters.csv", mime="text/csv")
        else:
            st.info("Clustering skipped — need ≥2 numeric cols & ≥20 rows.")
    with c2:
        slabel("ANOMALY DETECTION")
        if anomaly:
            st.metric("Anomalies detected", anomaly["count"])
            _fig(chart_pca(anomaly["pca_df"], "Anomaly Map (PCA 2D)"))
            slabel("ANOMALOUS RECORDS")
            anom_rows = anomaly["df"][anomaly["df"]["_status"]=="⚠️ Anomaly"].head(50)
            st.dataframe(anom_rows, use_container_width=True)
            st.download_button("⬇️ Download anomalies",
                               data=anom_rows.to_csv(index=False).encode(),
                               file_name="betterlytics_anomalies.csv", mime="text/csv")
        else:
            st.info("Anomaly detection skipped — need ≥2 numeric cols & ≥20 rows.")

# ── TEXT ──────────────────────────────────────
with t_text:
    text_data = scan.get("text")
    if not text_data:
        st.info("No text column found. Text analytics needs a column with varied, longer text entries.")
    else:
        txt_col = scan.get("text_col","")
        ib(f"Analyzing: <strong>{txt_col}</strong> &nbsp;·&nbsp; {text_data['n_docs']:,} docs &nbsp;·&nbsp; avg {text_data['avg_len']:.0f} chars")
        c1,c2 = st.columns(2)
        with c1:
            slabel("TOP TERMS (FREQUENCY)")
            _fig(chart_text_terms(text_data["top"]))
            st.dataframe(text_data["top"].head(20), use_container_width=True, hide_index=True)
        with c2:
            slabel("SENTIMENT")
            _fig(chart_sentiment(text_data["sentiment"]))
            st.dataframe(text_data["sentiment"], use_container_width=True, hide_index=True)
        if not text_data["tfidf"].empty:
            slabel("TF-IDF — DISTINCTIVE TERMS")
            ib("TF-IDF highlights terms that are frequent <em>and</em> distinctive — not just common filler words.")
            st.dataframe(text_data["tfidf"], use_container_width=True, hide_index=True)

# ── TIME SERIES ───────────────────────────────
with t_ts:
    ts = scan.get("ts")
    if not ts:
        st.info("No time-series analysis ran — need a datetime column + numeric column with ≥10 rows.")
    else:
        m1,m2,m3 = st.columns(3)
        m1.metric("Latest value", f"{ts['latest']:.4g}")
        m2.metric("Trend", ts["trend"])
        m3.metric("Recent change", f"{ts['pct']:+.1f}%")
        slabel("TIME SERIES CHART")
        _fig(chart_ts(ts))
        if ts.get("monthly") is not None:
            slabel("MONTHLY AVERAGE PATTERN")
            m = ts["monthly"]
            fig, ax = plt.subplots(figsize=(9,3.5))
            ax.bar(m["Month"],m[ts["val_col"]],color=CHART_COLORS[:len(m)],alpha=.85,width=.6)
            ax.set_title("Average by Month"); ax.set_ylabel(ts["val_col"]); fig.tight_layout()
            _fig(fig)

        slabel("INTERACTIVE DATE RANGE")
        dc, vc = ts["date_col"], ts["val_col"]
        if dc and vc and dc in df.columns and vc in df.columns:
            col_options = [c for c in schema.numeric if c != vc]
            vc2 = st.selectbox("Value column to chart", [vc]+col_options, key="ts_vc2")
            sub = df[[dc, vc2]].dropna().copy()
            sub[dc] = pd.to_datetime(sub[dc], errors="coerce")
            sub = sub.dropna().sort_values(dc)
            if not sub.empty:
                min_d, max_d = sub[dc].min().date(), sub[dc].max().date()
                d1,d2 = st.slider("Date range", min_value=min_d, max_value=max_d,
                                   value=(min_d, max_d), key="ts_range")
                mask = (sub[dc].dt.date >= d1) & (sub[dc].dt.date <= d2)
                sub2 = sub[mask]
                if not sub2.empty:
                    fig, ax = plt.subplots(figsize=(10,3.5))
                    ax.plot(sub2[dc], sub2[vc2], color=P["blue"], linewidth=1.5)
                    ax.set_title(f"{vc2} over time"); fig.autofmt_xdate(); fig.tight_layout()
                    _fig(fig)

# ── EDA TOOLS ─────────────────────────────────
with t_eda:
    slabel("DISTRIBUTION EXPLORER")
    if schema.numeric:
        max_cols = st.slider("Max columns to show", 1, min(12, len(schema.numeric)), min(6, len(schema.numeric)), key="dist_n")
        chosen_dist = st.multiselect("Columns", schema.numeric, default=schema.numeric[:max_cols], key="dist_cols")
        if chosen_dist:
            _fig(chart_distributions(df, chosen_dist))
    else:
        st.info("No numeric columns.")

    st.markdown("---")
    slabel("CATEGORY EXPLORER")
    if schema.categorical:
        cat_col = st.selectbox("Column", schema.categorical, key="cat_col")
        top_n_cat = st.slider("Top N values", 5, 50, 15, key="cat_n")
        _fig(chart_cat_bar(df, cat_col, top_n_cat))
        vc_df = df[cat_col].value_counts().head(top_n_cat).reset_index()
        vc_df.columns = ["Value","Count"]
        vc_df["% of total"] = (vc_df["Count"]/len(df)*100).round(2)
        st.dataframe(vc_df, use_container_width=True, hide_index=True)
    else:
        st.info("No categorical columns.")

    st.markdown("---")
    slabel("COLUMN STATISTICS")
    if schema.numeric:
        stat_cols = st.multiselect("Select columns", schema.numeric, default=schema.numeric[:6], key="stat_cols")
        if stat_cols:
            st.dataframe(df[stat_cols].describe(percentiles=[.05,.25,.5,.75,.95]).T.style.format(precision=4),
                         use_container_width=True)

    st.markdown("---")
    slabel("PIVOT TABLE")
    if schema.categorical and schema.numeric:
        pc1,pc2,pc3 = st.columns(3)
        piv_row  = pc1.selectbox("Row (index)", schema.categorical, key="piv_r")
        piv_col  = pc2.selectbox("Column", schema.categorical, index=min(1,len(schema.categorical)-1), key="piv_c")
        piv_val  = pc3.selectbox("Value", schema.numeric, key="piv_v")
        piv_agg  = st.selectbox("Aggregation", ["mean","sum","count","median","min","max"], key="piv_a")
        try:
            piv = df.pivot_table(index=piv_row, columns=piv_col, values=piv_val, aggfunc=piv_agg)
            st.dataframe(piv.style.format(precision=3).background_gradient(cmap="Blues", axis=None),
                         use_container_width=True)
        except Exception as e:
            st.warning(f"Could not build pivot: {e}")
    else:
        st.info("Need at least one categorical and one numeric column for pivot tables.")

    st.markdown("---")
    slabel("GROUP COMPARISON")
    if schema.categorical and schema.numeric:
        gc1,gc2 = st.columns(2)
        grp_col = gc1.selectbox("Group by", schema.categorical, key="grp_c")
        grp_val = gc2.selectbox("Measure", schema.numeric, key="grp_v")
        if df[grp_col].nunique() <= 30:
            grp = df.groupby(grp_col)[grp_val].agg(["mean","median","std","count"]).round(3)
            grp.columns = ["Mean","Median","Std Dev","Count"]
            st.dataframe(grp.sort_values("Mean",ascending=False), use_container_width=True)
            fig, ax = plt.subplots(figsize=(9,4))
            grp_sorted = grp.sort_values("Mean",ascending=False).head(20)
            ax.bar(grp_sorted.index, grp_sorted["Mean"],
                   color=CHART_COLORS[:len(grp_sorted)], alpha=.85)
            ax.set_xlabel(grp_col); ax.set_ylabel(f"Mean {grp_val}")
            ax.set_title(f"{grp_val} by {grp_col}")
            plt.xticks(rotation=45, ha="right", fontsize=8); fig.tight_layout()
            _fig(fig)
        else:
            st.info("Too many unique values to group (> 30). Pick a lower-cardinality column.")

# ── SCENARIO LAB ──────────────────────────────
with t_sim:
    slabel("WHAT-IF SCENARIO SIMULATOR")
    if not modeling or not modeling.get("best"):
        wb("Run a predictive model first — the Scenario Lab uses the trained model to simulate outcomes.")
    else:
        target_name = modeling["target"]
        ib(f"Predicting: <strong>{target_name}</strong> &nbsp;·&nbsp; Model: <strong>{modeling['best']}</strong>")
        edit_cols = [c for c in df.columns if c != target_name]
        num_edit  = [c for c in edit_cols if pd.api.types.is_numeric_dtype(df[c])]
        cat_edit  = [c for c in edit_cols if c in schema.categorical and df[c].nunique() <= 30]

        if not num_edit and not cat_edit:
            st.info("No editable features found.")
        else:
            row_idx = st.number_input("Row index to simulate",
                                      min_value=0, max_value=len(df)-1, value=0, step=1, key="sim_idx")
            base = df.iloc[int(row_idx)].copy()

            with st.expander("Current row values", expanded=False):
                st.dataframe(pd.DataFrame(base).T, use_container_width=True)

            st.markdown("**Adjust values:**")
            adjustments = {}
            if num_edit:
                chosen_num = st.multiselect("Numeric fields to change", num_edit,
                                            default=num_edit[:min(3,len(num_edit))], key="sim_num")
                if chosen_num:
                    ncols_sim = min(3, len(chosen_num))
                    cols_sim  = st.columns(ncols_sim)
                    for i, col in enumerate(chosen_num):
                        cur = float(base[col]) if pd.notna(base[col]) else 0.0
                        mn  = float(df[col].min())
                        mx  = float(df[col].max())
                        step = max(0.001, round((mx-mn)/200, 4))
                        with cols_sim[i % ncols_sim]:
                            adjustments[col] = st.slider(col, mn, mx, cur, step, key=f"sim_{col}")

            if cat_edit:
                chosen_cat = st.multiselect("Categorical fields to change", cat_edit, key="sim_cat")
                cat_row = st.columns(min(3, max(1, len(chosen_cat)))) if chosen_cat else []
                for i, col in enumerate(chosen_cat):
                    opts = sorted(df[col].dropna().unique().tolist())
                    cur_val = str(base[col]) if pd.notna(base[col]) else (opts[0] if opts else "")
                    cur_idx = opts.index(cur_val) if cur_val in opts else 0
                    with cat_row[i % len(cat_row)]:
                        adjustments[col] = st.selectbox(col, opts, index=cur_idx, key=f"sim_cat_{col}")

            if st.button("▶️ Run Simulation", type="primary"):
                result = scenario_sim(modeling, base, adjustments)
                if result and "error" not in result:
                    r1,r2,r3 = st.columns(3)
                    r1.metric("Original prediction", f"{result['original']}")
                    r2.metric("Adjusted prediction", f"{result['adjusted']}")
                    if result.get("delta") is not None:
                        r3.metric("Δ Change", f"{result['delta']:+.4f}")
                    gb("Simulation complete! The delta shows how much the prediction changed based on your adjustments.")
                elif result and "error" in result:
                    rb(f"Simulation error: {result['error']}")
                else:
                    wb("Could not run simulation for this row. Try a different row index.")

            st.markdown("---")
            slabel("BATCH SIMULATION")
            ib("Vary one numeric feature across a range to see how the prediction changes.")
            if num_edit:
                sweep_col = st.selectbox("Feature to sweep", num_edit, key="sweep_col")
                sweep_n   = st.slider("Number of steps", 5, 50, 20, key="sweep_n")
                if st.button("▶️ Run Sweep", key="sweep_btn"):
                    mn = float(df[sweep_col].min()); mx = float(df[sweep_col].max())
                    vals = np.linspace(mn, mx, sweep_n)
                    preds_sweep = []
                    for v in vals:
                        r = scenario_sim(modeling, base, {sweep_col: v})
                        preds_sweep.append(r["adjusted"] if r and "error" not in r else np.nan)
                    fig, ax = plt.subplots(figsize=(9,4))
                    ax.plot(vals, preds_sweep, color=P["blue"], linewidth=2, marker="o", markersize=4)
                    ax.axvline(float(base[sweep_col]), color=P["amber"], linewidth=1.5, linestyle="--", label="Current value")
                    ax.set_xlabel(sweep_col); ax.set_ylabel("Predicted outcome")
                    ax.set_title(f"Prediction vs {sweep_col}"); ax.legend(); fig.tight_layout()
                    _fig(fig)
                    sweep_df = pd.DataFrame({sweep_col: vals, "Predicted": preds_sweep})
                    st.dataframe(sweep_df.style.format(precision=4), use_container_width=True, hide_index=True)

# ── RUN LOG ───────────────────────────────────
with t_log:
    slabel("RUN LOG")
    ib("Every step reports status so you always know what ran, what was skipped, and why.")
    st.dataframe(pd.DataFrame(scan["log"]), use_container_width=True, hide_index=True)

# ── DOWNLOADS ─────────────────────────────────
st.markdown("---")
slabel("DOWNLOADS")
d1,d2,d3,d4 = st.columns(4)
d1.download_button("⬇️ Cleaned CSV",
    data=df.to_csv(index=False).encode(),
    file_name=f"betterlytics_{uploaded.name}", mime="text/csv", use_container_width=True)
d2.download_button("⬇️ Correlations CSV",
    data=corr_df.to_csv(index=False).encode() if corr_df is not None and not corr_df.empty else b"",
    file_name="correlations.csv", mime="text/csv", use_container_width=True)
if modeling and modeling.get("feat_imp") is not None:
    d3.download_button("⬇️ Feature Importance",
        data=modeling["feat_imp"].to_csv(index=False).encode(),
        file_name="feature_importance.csv", mime="text/csv", use_container_width=True)
d4.download_button("⬇️ Full Scan JSON",
    data=safe_json(scan).encode(),
    file_name="betterlytics_scan.json", mime="application/json", use_container_width=True)

st.markdown(f"""<div style="margin-top:3rem;text-align:center;color:{P['bg3']};
font-size:.72rem;font-family:'Space Mono',monospace">
betterlytics v5 · streamlit + scikit-learn
</div>""", unsafe_allow_html=True)
