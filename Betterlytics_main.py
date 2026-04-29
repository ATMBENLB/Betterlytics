from __future__ import annotations

import io
import json
import math
import re
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


st.set_page_config(page_title="betterlytics", layout="wide")

st.markdown(
    """
    <style>
    .main-title {font-size: 2.5rem; font-weight: 800; margin-bottom: 0.1rem;}
    .subtle {color: #667085; margin-bottom: 1rem;}
    .metric-card {padding: 0.8rem 1rem; border: 1px solid #e5e7eb; border-radius: 12px; background: #fafafa;}
    .section-note {padding: 0.8rem 1rem; background: #f8fafc; border-left: 4px solid #14b8a6; border-radius: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@dataclass
class Schema:
    numeric: list[str]
    categorical: list[str]
    text: list[str]
    datetime: list[str]
    boolean: list[str]
    target_candidates: list[str]


@st.cache_data(show_spinner=False)
def load_csv(raw_bytes: bytes) -> pd.DataFrame:
    attempts = [
        {"sep": None, "engine": "python", "encoding": "utf-8"},
        {"sep": ",", "engine": "python", "encoding": "utf-8"},
        {"sep": None, "engine": "python", "encoding": "latin-1"},
    ]
    last_exc = None
    for kwargs in attempts:
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), **kwargs)
            if df.shape[1] >= 1:
                break
        except Exception as exc:
            last_exc = exc
    else:
        raise last_exc or ValueError("Could not read CSV")

    df.columns = [clean_column_name(c) for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"": np.nan, "nan": np.nan, "None": np.nan, "null": np.nan})
    return auto_parse_datetimes(df)


def clean_column_name(name: Any) -> str:
    text = str(name).strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^0-9a-zA-Z_]+", "", text)
    return text or "unnamed_column"



def auto_parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype != object:
            continue
        series = out[col].dropna().astype(str)
        if series.empty:
            continue
        sample = series.head(100)
        parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
        if parsed.notna().mean() >= 0.8:
            full = pd.to_datetime(out[col], errors="coerce", infer_datetime_format=True)
            if full.notna().mean() >= 0.7:
                out[col] = full
    return out



def infer_schema(df: pd.DataFrame) -> Schema:
    numeric = []
    categorical = []
    text = []
    datetime_cols = []
    boolean = []

    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            boolean.append(col)
        elif pd.api.types.is_datetime64_any_dtype(s):
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            numeric.append(col)
        else:
            nunique = s.nunique(dropna=True)
            avg_len = s.dropna().astype(str).str.len().mean() if s.dropna().shape[0] else 0
            if avg_len > 30 or nunique > max(25, min(200, int(len(df) * 0.3))):
                text.append(col)
            else:
                categorical.append(col)

    target_candidates = []
    for col in numeric:
        nunique = df[col].nunique(dropna=True)
        if nunique >= 5:
            target_candidates.append(col)
    for col in categorical + boolean:
        nunique = df[col].nunique(dropna=True)
        if 2 <= nunique <= min(20, max(2, int(len(df) * 0.2))):
            target_candidates.append(col)

    return Schema(
        numeric=numeric,
        categorical=categorical,
        text=text,
        datetime=datetime_cols,
        boolean=boolean,
        target_candidates=target_candidates,
    )



def basic_profile(df: pd.DataFrame) -> dict[str, Any]:
    categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c])]
    return {
        "shape": df.shape,
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1_000_000),
        "missing": df.isna().sum().sort_values(ascending=False),
        "missing_pct": (df.isna().mean() * 100).sort_values(ascending=False),
        "numeric_summary": df.describe(include=[np.number]).T if len(df.select_dtypes(include=[np.number]).columns) else pd.DataFrame(),
        "categorical_summary": (
            pd.DataFrame({
                "unique_values": df[categorical_cols].nunique(),
                "top_example": [df[c].mode(dropna=True).iloc[0] if not df[c].mode(dropna=True).empty else np.nan for c in categorical_cols],
            }) if categorical_cols else pd.DataFrame()
        ),
    }



def dataset_health_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "missing_pct": round(float(df[col].isna().mean() * 100), 2),
            "unique_values": int(df[col].nunique(dropna=True)),
            "sample": None if df[col].dropna().empty else str(df[col].dropna().iloc[0])[:60],
        })
    return pd.DataFrame(rows).sort_values(["missing_pct", "unique_values"], ascending=[False, False])



def target_recommendations(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    rows = []
    keywords = ["target", "label", "outcome", "response", "sales", "revenue", "conversion", "profit", "price", "score", "churn"]
    for col in schema.target_candidates:
        score = 0.0
        reasons = []
        if any(k in col.lower() for k in keywords):
            score += 2.5
            reasons.append("name looks outcome-like")
        missing = df[col].isna().mean()
        if missing < 0.1:
            score += 1.5
            reasons.append("low missingness")
        if col in schema.numeric:
            nunique = df[col].nunique(dropna=True)
            if nunique >= 10:
                score += 1.5
                reasons.append("good numeric spread")
        else:
            nunique = df[col].nunique(dropna=True)
            if 2 <= nunique <= 10:
                score += 1.5
                reasons.append("good class count")
        rows.append({"column": col, "score": round(score, 2), "why": ", ".join(reasons)})
    out = pd.DataFrame(rows)
    return out.sort_values(["score", "column"], ascending=[False, True]).reset_index(drop=True) if not out.empty else out



def strongest_correlations(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return pd.DataFrame(columns=["feature_1", "feature_2", "correlation", "strength"])
    corr = num.corr(numeric_only=True)
    pairs = []
    cols = corr.columns.tolist()
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            value = corr.loc[c1, c2]
            if pd.notna(value):
                strength = abs(value)
                pairs.append({
                    "feature_1": c1,
                    "feature_2": c2,
                    "correlation": round(float(value), 4),
                    "strength": round(float(strength), 4),
                })
    out = pd.DataFrame(pairs).sort_values("strength", ascending=False).head(top_n)
    return out



def recommend_text_column(df: pd.DataFrame, schema: Schema) -> str | None:
    candidates = schema.text + [c for c in schema.categorical if df[c].dropna().astype(str).str.len().mean() > 20]
    return candidates[0] if candidates else None



def recommend_date_value_columns(schema: Schema) -> tuple[str | None, str | None]:
    date_col = schema.datetime[0] if schema.datetime else None
    value_col = schema.numeric[0] if schema.numeric else None
    return date_col, value_col



def make_preprocessor(df: pd.DataFrame, target: str | None) -> tuple[ColumnTransformer, list[str], list[str]]:
    features = [c for c in df.columns if c != target]
    X = df[features].copy()
    numeric_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in X.columns if c not in numeric_features and not pd.api.types.is_datetime64_any_dtype(X[c])]
    datetime_features = [c for c in X.columns if pd.api.types.is_datetime64_any_dtype(X[c])]

    for c in datetime_features:
        X[c + "_year"] = X[c].dt.year
        X[c + "_month"] = X[c].dt.month
        X[c + "_day"] = X[c].dt.day
        X[c + "_dow"] = X[c].dt.dayofweek
    X = X.drop(columns=datetime_features, errors="ignore")

    numeric_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_features, categorical_features



def detect_problem_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "classification" if series.nunique(dropna=True) <= 10 else "regression"
    return "classification"



def model_library(problem_type: str) -> dict[str, Any]:
    if problem_type == "regression":
        return {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=6),
            "Random Forest": RandomForestRegressor(random_state=42, n_estimators=200, max_depth=10),
            "Boosted Trees": GradientBoostingRegressor(random_state=42),
            "Support Vector Regression": SVR(),
        }
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=6),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200, max_depth=10),
        "Boosted Trees": GradientBoostingClassifier(random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42),
    }



def auto_optimize_models(problem_type: str, n_rows: int, n_numeric: int) -> list[str]:
    if problem_type == "regression":
        picks = ["Linear Regression", "Random Forest", "Boosted Trees"]
        if n_rows < 300:
            picks = ["Linear Regression", "Decision Tree"]
        if n_numeric <= 2:
            picks = ["Linear Regression", "Random Forest"]
        return picks
    picks = ["Logistic Regression", "Random Forest", "Boosted Trees"]
    if n_rows < 300:
        picks = ["Logistic Regression", "Decision Tree"]
    if n_numeric <= 2:
        picks = ["Logistic Regression", "Random Forest"]
    return picks



def train_models(df: pd.DataFrame, target: str, chosen_models: list[str], test_size: float = 0.2) -> dict[str, Any]:
    y = df[target]
    problem_type = detect_problem_type(y)
    clean = df.dropna(subset=[target]).copy()
    y = clean[target]
    preprocessor, _, _ = make_preprocessor(clean, target)
    X = clean.drop(columns=[target])

    if problem_type == "classification":
        stratify = y if y.nunique(dropna=True) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    library = model_library(problem_type)
    results = []
    plots = {}

    for name in chosen_models:
        model = library[name]
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        record = {"model": name}
        if problem_type == "regression":
            record["R2"] = float(r2_score(y_test, preds))
            record["RMSE"] = float(math.sqrt(mean_squared_error(y_test, preds)))
            record["MAE"] = float(mean_absolute_error(y_test, preds))
            record["score_for_ranking"] = record["R2"]
        else:
            record["Accuracy"] = float(accuracy_score(y_test, preds))
            record["score_for_ranking"] = record["Accuracy"]
            record["report"] = classification_report(y_test, preds, output_dict=True, zero_division=0)
            record["confusion_matrix"] = confusion_matrix(y_test, preds)
        results.append(record)
        plots[name] = {"y_test": y_test, "preds": preds, "pipeline": pipe}

    leaderboard = pd.DataFrame(results).sort_values("score_for_ranking", ascending=False).reset_index(drop=True)
    best_name = leaderboard.iloc[0]["model"] if not leaderboard.empty else None
    return {
        "problem_type": problem_type,
        "leaderboard": leaderboard,
        "details": plots,
        "best_model": best_name,
    }



def run_kmeans_scan(df: pd.DataFrame, max_k: int = 6) -> dict[str, Any] | None:
    num = df.select_dtypes(include=[np.number]).copy()
    if num.shape[1] < 2 or len(num) < 20:
        return None
    num = num.fillna(num.median(numeric_only=True))
    scaler = StandardScaler()
    X = scaler.fit_transform(num)
    best = None
    for k in range(2, min(max_k, len(num) - 1) + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        if best is None or score > best["silhouette"]:
            best = {"k": k, "silhouette": float(score), "labels": labels, "model": model}
    if best is None:
        return None
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    pca_df = pd.DataFrame({"pc1": coords[:, 0], "pc2": coords[:, 1], "cluster": best["labels"]})
    clustered_df = df.copy()
    clustered_df["cluster"] = best["labels"]
    return {"k": best["k"], "silhouette": best["silhouette"], "pca_df": pca_df, "clustered_df": clustered_df}



def run_anomaly_scan(df: pd.DataFrame) -> dict[str, Any] | None:
    num = df.select_dtypes(include=[np.number]).copy()
    if num.shape[1] < 2 or len(num) < 20:
        return None
    num = num.fillna(num.median(numeric_only=True))
    scaler = StandardScaler()
    X = scaler.fit_transform(num)
    model = IsolationForest(contamination=0.05, random_state=42)
    preds = model.fit_predict(X)
    scores = model.decision_function(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    out = df.copy()
    out["anomaly_flag"] = np.where(preds == -1, "Likely anomaly", "Normal")
    out["anomaly_score"] = scores
    pca_df = pd.DataFrame({"pc1": coords[:, 0], "pc2": coords[:, 1], "cluster": out["anomaly_flag"]})
    return {"anomaly_count": int((preds == -1).sum()), "scored_df": out.sort_values("anomaly_score"), "pca_df": pca_df}



def run_text_scan(df: pd.DataFrame, text_col: str) -> dict[str, Any] | None:
    series = df[text_col].dropna().astype(str)
    if series.empty:
        return None
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=25)
    matrix = vectorizer.fit_transform(series)
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    vocab = np.array(vectorizer.get_feature_names_out())
    top = pd.DataFrame({"term": vocab, "count": counts}).sort_values("count", ascending=False).reset_index(drop=True)
    positive_words = {"good", "great", "excellent", "love", "best", "easy", "fast", "strong", "happy", "success"}
    negative_words = {"bad", "poor", "slow", "hard", "hate", "worse", "worst", "issue", "problem", "fail"}
    sentiments = []
    for text in series.head(5000):
        words = re.findall(r"[a-zA-Z']+", text.lower())
        pos = sum(1 for w in words if w in positive_words)
        neg = sum(1 for w in words if w in negative_words)
        sentiments.append("Positive" if pos > neg else "Negative" if neg > pos else "Neutral")
    sent_df = pd.DataFrame({"sentiment": sentiments})
    return {"top_terms": top.head(20), "sentiment": sent_df["sentiment"].value_counts().reset_index(name="count").rename(columns={"index": "sentiment"})}



def run_time_series_scan(df: pd.DataFrame, date_col: str, value_col: str) -> dict[str, Any] | None:
    temp = df[[date_col, value_col]].dropna().copy()
    if temp.empty:
        return None
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna().sort_values(date_col)
    if len(temp) < 10:
        return None
    ts = temp.groupby(date_col)[value_col].mean().reset_index()
    ts["rolling_mean"] = ts[value_col].rolling(window=min(7, max(2, len(ts)//5)), min_periods=1).mean()
    latest = float(ts[value_col].iloc[-1])
    prior = float(ts[value_col].iloc[max(0, len(ts)-4):-1].mean()) if len(ts) > 3 else float(ts[value_col].mean())
    trend = "up" if latest > prior else "down" if latest < prior else "flat"
    return {"series": ts, "trend": trend, "latest": latest, "reference": prior}



def make_corr_plot(df: pd.DataFrame):
    num = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(8, 5))
    if num.shape[1] < 2:
        ax.text(0.5, 0.5, "Need at least 2 numeric columns", ha="center", va="center")
        ax.axis("off")
        return fig
    corr = num.corr(numeric_only=True)
    im = ax.imshow(corr, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig



def make_missing_plot(df: pd.DataFrame):
    miss = (df.isna().mean() * 100).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 4))
    if miss.empty:
        ax.text(0.5, 0.5, "No columns found", ha="center", va="center")
        ax.axis("off")
        return fig
    miss.plot(kind="bar", ax=ax)
    ax.set_ylabel("Missing %")
    ax.set_title("Top missing-value columns")
    fig.tight_layout()
    return fig



def make_cluster_plot(pca_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, subset in pca_df.groupby("cluster"):
        ax.scatter(subset["pc1"], subset["pc2"], label=str(label), alpha=0.7)
    ax.set_title("2D map of groups")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    fig.tight_layout()
    return fig



def make_time_plot(ts: pd.DataFrame, date_col: str, value_col: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ts[date_col], ts[value_col], label="Actual")
    ax.plot(ts[date_col], ts["rolling_mean"], label="Smoothed trend")
    ax.set_title("Time trend")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig



def create_narrative(scan: dict[str, Any]) -> str:
    lines = []
    profile = scan["profile"]
    lines.append(f"This dataset has {profile['shape'][0]:,} rows and {profile['shape'][1]} columns.")
    if scan["health_flags"]:
        lines.append("Data quality watchouts: " + "; ".join(scan["health_flags"]) + ".")
    corr = scan.get("correlations")
    if corr is not None and not corr.empty:
        top = corr.iloc[0]
        lines.append(
            f"The strongest numeric relationship is between {top['feature_1']} and {top['feature_2']} with a correlation of {top['correlation']:.2f}. "
            "That suggests a potentially meaningful connection worth validating in context."
        )
    modeling = scan.get("modeling")
    if modeling and not modeling["leaderboard"].empty:
        leader = modeling["leaderboard"].iloc[0]
        if modeling["problem_type"] == "regression":
            lines.append(
                f"For prediction, the best-performing regression model was {leader['model']} with an R2 of {leader['R2']:.2f}. "
                "Use this as directional evidence rather than a final production score until you validate on new data."
            )
        else:
            lines.append(
                f"For classification, the strongest model was {leader['model']} with an accuracy of {leader['Accuracy']:.2f}. "
                "This means the app found a useful signal pattern, though class balance and business cost of mistakes still matter."
            )
    clustering = scan.get("clustering")
    if clustering:
        lines.append(
            f"The data naturally separates into about {clustering['k']} groups with a silhouette score of {clustering['silhouette']:.2f}. "
            "That is a useful starting point for segmentation if the groups also make business sense."
        )
    anomaly = scan.get("anomaly")
    if anomaly:
        lines.append(f"The anomaly scan flagged {anomaly['anomaly_count']} records that look meaningfully different from the rest.")
    text = scan.get("text")
    if text:
        top_term = text["top_terms"].iloc[0]["term"] if not text["top_terms"].empty else None
        if top_term:
            lines.append(f"In the text field, the most repeated term pattern was '{top_term}', which may reflect a dominant topic or customer concern.")
    ts = scan.get("time_series")
    if ts:
        lines.append(f"The time-series view shows the tracked metric is trending {ts['trend']} lately.")
    lines.append("Overall, betterlytics is surfacing the strongest patterns first so a beginner can focus on the highest-value questions before digging deeper.")
    return "\n\n".join(lines)





@dataclass
class Decision:
    title: str
    category: str
    priority: str
    confidence: float
    estimated_impact: str
    rationale: str
    risk: str
    action_steps: list[str]


def _priority_from_score(score: float) -> str:
    if score >= 8:
        return "High"
    if score >= 5:
        return "Medium"
    return "Low"


def _confidence_label(conf: float) -> str:
    if conf >= 0.8:
        return "High"
    if conf >= 0.6:
        return "Moderate"
    return "Low"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def prettify_feature_name(name: str) -> str:
    text = str(name)
    text = re.sub(r"^(num|cat)__", "", text)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_model_drivers(scan: dict[str, Any], top_n: int = 10) -> pd.DataFrame:
    modeling = scan.get("modeling")
    if not modeling:
        return pd.DataFrame(columns=["feature", "importance"])

    best_name = modeling.get("best_model")
    details = modeling.get("details", {})
    best = details.get(best_name)
    if not best:
        return pd.DataFrame(columns=["feature", "importance"])

    pipe = best.get("pipeline")
    if pipe is None:
        return pd.DataFrame(columns=["feature", "importance"])

    try:
        prep = pipe.named_steps["prep"]
        model = pipe.named_steps["model"]
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])

    try:
        feature_names = [prettify_feature_name(x) for x in prep.get_feature_names_out()]
    except Exception:
        feature_names = None

    if feature_names is None:
        return pd.DataFrame(columns=["feature", "importance"])

    values = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        values = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)

    if values is None:
        return pd.DataFrame(columns=["feature", "importance"])

    out = pd.DataFrame({"feature": feature_names, "importance": values})
    out["abs_importance"] = out["importance"].abs()
    out = out.groupby("feature", as_index=False)[["importance", "abs_importance"]].sum()
    out = out.sort_values("abs_importance", ascending=False).head(top_n).drop(columns="abs_importance")
    return out.reset_index(drop=True)


def build_cluster_profiles(scan: dict[str, Any], top_n_numeric: int = 5) -> pd.DataFrame:
    clustering = scan.get("clustering")
    if not clustering or "clustered_df" not in clustering:
        return pd.DataFrame()

    df = clustering["clustered_df"].copy()
    if "cluster" not in df.columns:
        return pd.DataFrame()

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "cluster"]
    if not numeric_cols:
        return pd.DataFrame()

    overall_means = df[numeric_cols].mean(numeric_only=True)
    rows = []
    for cluster_id, subset in df.groupby("cluster"):
        cluster_means = subset[numeric_cols].mean(numeric_only=True)
        deltas = (cluster_means - overall_means).sort_values(key=lambda s: s.abs(), ascending=False)
        defining = deltas.head(top_n_numeric)
        rows.append({
            "cluster": cluster_id,
            "size": len(subset),
            "share_pct": round(len(subset) / len(df) * 100, 2),
            "defining_features": ", ".join([f"{idx} ({val:+.2f})" for idx, val in defining.items()]),
        })
    return pd.DataFrame(rows).sort_values("share_pct", ascending=False).reset_index(drop=True)


def scenario_simulator(scan: dict[str, Any], base_row: pd.Series | dict[str, Any], adjustments: dict[str, Any]) -> dict[str, Any] | None:
    modeling = scan.get("modeling")
    if not modeling:
        return None

    best_name = modeling.get("best_model")
    details = modeling.get("details", {})
    best = details.get(best_name)
    if not best:
        return None

    pipe = best.get("pipeline")
    if pipe is None:
        return None

    row = base_row.to_dict() if isinstance(base_row, pd.Series) else dict(base_row)
    original = row.copy()
    for key, value in adjustments.items():
        if key in row:
            row[key] = value

    original_df = pd.DataFrame([original])
    adjusted_df = pd.DataFrame([row])

    try:
        original_pred = pipe.predict(original_df)[0]
        adjusted_pred = pipe.predict(adjusted_df)[0]
    except Exception:
        return None

    delta = None
    if isinstance(original_pred, (int, float, np.number)) and isinstance(adjusted_pred, (int, float, np.number)):
        delta = float(adjusted_pred) - float(original_pred)

    return {
        "model": best_name,
        "original_prediction": original_pred.item() if hasattr(original_pred, "item") else original_pred,
        "adjusted_prediction": adjusted_pred.item() if hasattr(adjusted_pred, "item") else adjusted_pred,
        "delta": delta,
        "adjustments": adjustments,
    }


def create_decision_summary(decision_df: pd.DataFrame, opportunities: list[str], risks: list[str]) -> str:
    lines = []
    if not decision_df.empty:
        top = decision_df.iloc[0]
        lines.append(
            f"The highest-priority action is: {top['title']}. Confidence is {top['confidence_label'].lower()} ({top['confidence']:.2f})."
        )
    if opportunities:
        lines.append("Biggest opportunities: " + "; ".join(opportunities[:3]) + ".")
    if risks:
        lines.append("Main risks to manage: " + "; ".join(risks[:3]) + ".")
    if not decision_df.empty:
        lines.append("This layer translates the scan into ranked decisions so users can act faster.")
    return "\n\n".join(lines)


def generate_decision_engine(scan: dict[str, Any]) -> dict[str, Any]:
    decisions: list[Decision] = []
    risks: list[str] = []
    opportunities: list[str] = []

    corr = scan.get("correlations")
    modeling = scan.get("modeling")
    clustering = scan.get("clustering")
    anomaly = scan.get("anomaly")
    time_series = scan.get("time_series")
    for flag in scan.get("health_flags", []):
        risks.append(flag)

    drivers = extract_model_drivers(scan, top_n=8)

    if modeling is not None and not modeling["leaderboard"].empty:
        leader = modeling["leaderboard"].iloc[0]
        problem_type = modeling["problem_type"]
        model_name = leader["model"]
        top_driver = drivers.iloc[0]["feature"] if not drivers.empty else None

        if problem_type == "classification":
            acc = _safe_float(leader.get("Accuracy"))
            conf = min(0.95, max(0.3, acc))
            score = acc * 10
            rationale = f"The best classification model reached {acc:.2f} accuracy, suggesting useful predictive signal."
            if top_driver:
                rationale += f" The strongest modeled driver appears to be {top_driver}."
            decisions.append(Decision(
                title=f"Use {model_name} as a decision-support scoring layer",
                category="Predictive modeling",
                priority=_priority_from_score(score),
                confidence=conf,
                estimated_impact="Can improve prioritization, routing, targeting, or risk flagging after validation.",
                rationale=rationale,
                risk="Review class balance and business costs of false positives and false negatives before automation.",
                action_steps=[
                    "Validate on fresh data or a later time window",
                    "Inspect misclassifications by business impact",
                    "Use scores for ranking before full automation",
                ],
            ))
            opportunities.append("Predictive classification signal is strong enough for prioritization experiments.")
        else:
            r2 = _safe_float(leader.get("R2"))
            conf = min(0.95, max(0.3, 0.5 + (r2 * 0.5)))
            score = max(0.0, r2) * 10
            rationale = f"The top regression model achieved an R2 of {r2:.2f}, so a meaningful share of variation is explainable."
            if top_driver:
                rationale += f" The strongest modeled driver appears to be {top_driver}."
            decisions.append(Decision(
                title=f"Use {model_name} for directional forecasting and what-if planning",
                category="Forecasting",
                priority=_priority_from_score(score),
                confidence=conf,
                estimated_impact="Can improve planning, budgeting, and scenario analysis if the target is operationally meaningful.",
                rationale=rationale,
                risk="Treat this as directional guidance until it is stable across time and new data.",
                action_steps=[
                    "Compare predictions against newer data",
                    "Stress-test edge cases",
                    "Use top features in scenario planning",
                ],
            ))
            opportunities.append("Forecasting signal exists and can be converted into scenario planning.")

    if corr is not None and not corr.empty:
        top_corr = corr.iloc[0]
        strength = _safe_float(top_corr["strength"])
        if strength >= 0.6:
            signed_corr = _safe_float(top_corr["correlation"])
            f1 = top_corr["feature_1"]
            f2 = top_corr["feature_2"]
            direction = "move together" if signed_corr > 0 else "move in opposite directions"
            decisions.append(Decision(
                title=f"Investigate the {f1} / {f2} relationship",
                category="Driver analysis",
                priority=_priority_from_score(strength * 10),
                confidence=min(0.9, 0.45 + strength * 0.5),
                estimated_impact="Potentially high if one variable is controllable and the relationship is real.",
                rationale=f"The strongest numeric relationship is between {f1} and {f2} (correlation {signed_corr:.2f}), suggesting they meaningfully {direction}.",
                risk="This may reflect leakage, coincidence, or shared dependence on another variable.",
                action_steps=[
                    "Check for leakage or duplicate information",
                    "Test subgroup stability",
                    "Determine whether one variable is actionable",
                ],
            ))
            opportunities.append(f"A strong relationship exists between {f1} and {f2}.")

    if clustering:
        k = clustering.get("k")
        silhouette = _safe_float(clustering.get("silhouette"))
        if k and silhouette >= 0.2:
            decisions.append(Decision(
                title=f"Operationalize the {k} discovered segments",
                category="Segmentation",
                priority=_priority_from_score(silhouette * 20),
                confidence=min(0.9, 0.4 + silhouette),
                estimated_impact="Can improve targeting, pricing, retention, and differentiated offers if the groups map to real behavior.",
                rationale=f"The clustering scan found {k} groups with a silhouette score of {silhouette:.2f}, indicating potentially usable segment structure.",
                risk="Mathematical clusters are not automatically business segments; profile them before acting.",
                action_steps=[
                    "Profile each segment in business terms",
                    "Compare outcomes by segment",
                    "Test differentiated strategies",
                ],
            ))
            opportunities.append("The dataset appears segmentable enough for differentiated strategies.")

    if anomaly:
        anomaly_count = int(anomaly.get("anomaly_count", 0))
        if anomaly_count > 0:
            decisions.append(Decision(
                title="Review anomalous records before scaling decisions",
                category="Risk control",
                priority=_priority_from_score(min(10, 4 + anomaly_count / 5)),
                confidence=0.75,
                estimated_impact="Can prevent distorted models, bad reporting, and missed high-value edge cases.",
                rationale=f"The anomaly scan flagged {anomaly_count} materially unusual records.",
                risk="Anomalies can be data errors, rare events, fraud, or valuable exceptions.",
                action_steps=[
                    "Inspect the highest-risk anomalies manually",
                    "Separate data errors from true edge cases",
                    "Decide whether to exclude, cap, or monitor them separately",
                ],
            ))
            risks.append(f"{anomaly_count} anomalous records warrant manual review.")

    if time_series:
        trend = time_series.get("trend", "flat")
        latest = _safe_float(time_series.get("latest"))
        reference = _safe_float(time_series.get("reference"))
        if trend in {"up", "down"}:
            decisions.append(Decision(
                title=f"Respond to the current {trend} trend in the tracked metric",
                category="Time-series monitoring",
                priority="High" if trend == "down" else "Medium",
                confidence=0.7,
                estimated_impact="Supports faster reaction before directional changes compound.",
                rationale=f"The time-series module shows the metric trending {trend}. Latest value: {latest:.2f}. Recent reference level: {reference:.2f}.",
                risk="Short-term movement can be noisy and may not represent a durable trend.",
                action_steps=[
                    "Validate the trend over a longer window",
                    "Check which subgroups are driving the movement",
                    "Set thresholds for alerts",
                ],
            ))

    if not decisions:
        decisions.append(Decision(
            title="Improve data quality and define a stronger outcome variable",
            category="Foundation",
            priority="High",
            confidence=0.8,
            estimated_impact="Creates the conditions for stronger modeling and clearer decisions later.",
            rationale="The current scan is more exploratory than decision-grade.",
            risk="Acting too early may overfit noise or incomplete structure.",
            action_steps=[
                "Reduce missingness in key fields",
                "Clarify target definitions",
                "Collect richer features or more rows",
            ],
        ))

    decision_df = pd.DataFrame([{
        "title": d.title,
        "category": d.category,
        "priority": d.priority,
        "confidence": round(d.confidence, 2),
        "confidence_label": _confidence_label(d.confidence),
        "estimated_impact": d.estimated_impact,
        "rationale": d.rationale,
        "risk": d.risk,
        "action_steps": " | ".join(d.action_steps),
    } for d in decisions])
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    decision_df["priority_rank"] = decision_df["priority"].map(priority_order).fillna(9)
    decision_df = decision_df.sort_values(["priority_rank", "confidence"], ascending=[True, False]).drop(columns="priority_rank").reset_index(drop=True)

    return {
        "decisions": decisions,
        "decision_df": decision_df,
        "drivers": drivers,
        "cluster_profiles": build_cluster_profiles(scan),
        "opportunities": opportunities,
        "risks": risks,
        "summary": create_decision_summary(decision_df, opportunities, risks),
    }

def create_implications(scan: dict[str, Any]) -> list[str]:
    actions = []
    modeling = scan.get("modeling")
    corr = scan.get("correlations")
    if modeling and not modeling["leaderboard"].empty:
        leader = modeling["leaderboard"].iloc[0]
        if modeling["problem_type"] == "classification" and leader["Accuracy"] >= 0.75:
            actions.append("You may have enough signal to support lead scoring, churn flagging, or conversion prediction experiments.")
        if modeling["problem_type"] == "regression" and leader["R2"] >= 0.5:
            actions.append("A meaningful share of outcome variation is explainable, so forecasting or pricing-response analysis may be worth pursuing.")
    if corr is not None and not corr.empty and corr.iloc[0]["strength"] >= 0.6:
        actions.append("At least one strong variable relationship exists, so driver analysis and multicollinearity checks should be part of the next pass.")
    clustering = scan.get("clustering")
    if clustering and clustering["silhouette"] >= 0.25:
        actions.append("The segment structure is strong enough to test differentiated messaging, offers, or retention strategies by group.")
    if not actions:
        actions.append("This dataset is better suited for exploration than immediate decision automation; focus first on cleaning, feature design, and business framing.")
    return actions



def health_flags(df: pd.DataFrame, schema: Schema) -> list[str]:
    flags = []
    if df.duplicated().sum() > 0:
        flags.append(f"{int(df.duplicated().sum())} duplicate rows found")
    if float(df.isna().mean().mean()) > 0.15:
        flags.append("overall missingness is elevated")
    if len(schema.numeric) < 2:
        flags.append("very limited numeric signal")
    if len(df) < 50:
        flags.append("small sample size")
    return flags



def auto_module_recommendations(df: pd.DataFrame, schema: Schema) -> dict[str, bool]:
    target_df = target_recommendations(df, schema)
    target_exists = not target_df.empty
    text_exists = recommend_text_column(df, schema) is not None
    date_col, value_col = recommend_date_value_columns(schema)
    return {
        "eda": True,
        "correlation": len(schema.numeric) >= 2,
        "modeling": target_exists,
        "clustering": len(schema.numeric) >= 2 and len(df) >= 20,
        "anomaly": len(schema.numeric) >= 2 and len(df) >= 20,
        "text": text_exists,
        "time_series": bool(date_col and value_col and len(df) >= 10),
    }



def run_total_scan(df: pd.DataFrame, schema: Schema, auto_optimize: bool, target_override: str | None = None, test_size: float = 0.2):
    progress = st.progress(0, text="Starting total analytical scan...")
    run_log = []
    scan = {
        "profile": basic_profile(df),
        "health_flags": health_flags(df, schema),
    }
    recommendations = auto_module_recommendations(df, schema)

    progress.progress(10, text="Profiling data quality and structure...")
    run_log.append({"step": "Profile", "status": "Completed"})

    progress.progress(25, text="Finding strongest correlations...")
    scan["correlations"] = strongest_correlations(df)
    run_log.append({"step": "Correlations", "status": "Completed" if not scan["correlations"].empty else "Skipped", "reason": "Need at least 2 numeric columns" if scan["correlations"].empty else ""})

    target_df = target_recommendations(df, schema)
    target = target_override or (target_df.iloc[0]["column"] if not target_df.empty else None)
    scan["target_recommendations"] = target_df

    if recommendations["modeling"] and target:
        progress.progress(45, text=f"Training the most useful predictive models for {target}...")
        problem_type = detect_problem_type(df[target])
        chosen_models = auto_optimize_models(problem_type, len(df), len(schema.numeric)) if auto_optimize else list(model_library(problem_type).keys())[:3]
        try:
            scan["modeling"] = train_models(df, target, chosen_models, test_size=test_size)
            scan["modeling"]["target"] = target
            run_log.append({"step": "Modeling", "status": "Completed", "target": target, "models": ", ".join(chosen_models)})
        except Exception as exc:
            scan["modeling"] = None
            run_log.append({"step": "Modeling", "status": "Failed", "reason": str(exc)})
    else:
        scan["modeling"] = None
        run_log.append({"step": "Modeling", "status": "Skipped", "reason": "No good target candidate found"})

    if recommendations["clustering"]:
        progress.progress(60, text="Looking for natural groups in the data...")
        scan["clustering"] = run_kmeans_scan(df)
        run_log.append({"step": "Clustering", "status": "Completed" if scan["clustering"] else "Skipped"})
    else:
        scan["clustering"] = None
        run_log.append({"step": "Clustering", "status": "Skipped", "reason": "Need more numeric columns and rows"})

    if recommendations["anomaly"]:
        progress.progress(72, text="Checking for unusual records worth investigation...")
        scan["anomaly"] = run_anomaly_scan(df)
        run_log.append({"step": "Anomaly detection", "status": "Completed" if scan["anomaly"] else "Skipped"})
    else:
        scan["anomaly"] = None
        run_log.append({"step": "Anomaly detection", "status": "Skipped", "reason": "Need more numeric columns and rows"})

    text_col = recommend_text_column(df, schema)
    if recommendations["text"] and text_col:
        progress.progress(84, text=f"Scanning text patterns in {text_col}...")
        scan["text"] = run_text_scan(df, text_col)
        scan["text_col"] = text_col
        run_log.append({"step": "Text analytics", "status": "Completed" if scan["text"] else "Skipped", "column": text_col})
    else:
        scan["text"] = None
        run_log.append({"step": "Text analytics", "status": "Skipped", "reason": "No useful text column detected"})

    date_col, value_col = recommend_date_value_columns(schema)
    if recommendations["time_series"] and date_col and value_col:
        progress.progress(94, text=f"Analyzing time trend using {date_col} and {value_col}...")
        scan["time_series"] = run_time_series_scan(df, date_col, value_col)
        scan["date_col"] = date_col
        scan["value_col"] = value_col
        run_log.append({"step": "Time-series", "status": "Completed" if scan["time_series"] else "Skipped", "date": date_col, "value": value_col})
    else:
        scan["time_series"] = None
        run_log.append({"step": "Time-series", "status": "Skipped", "reason": "Need date and numeric value columns"})

    progress.progress(100, text="Building dashboard and interpretation summary...")
    scan["narrative"] = create_narrative(scan)
    scan["implications"] = create_implications(scan)
    scan["run_log"] = run_log
    return scan



def safe_json(scan: dict[str, Any]) -> str:
    def convert(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return str(obj)
    return json.dumps(scan, default=convert, indent=2)


st.markdown('<div class="main-title">betterlytics</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">A beginner-friendly analytics engine that explains what it is doing, surfaces the strongest patterns, and translates results into plain English.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("1) Upload your data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], help="Upload one CSV. betterlytics will inspect the structure and suggest the best analysis path.")

    st.header("2) Beginner-friendly settings")
    auto_optimize = st.checkbox("Auto-optimize for maximum insight", value=True, help="When turned on, the app picks the most useful analyses and strongest starter models for your dataset.")
    test_size = st.slider("Test data share for model checking", 0.1, 0.4, 0.2, 0.05, help="A small piece of the data is held out to see how well a model generalizes.")

if uploaded_file is None:
    st.info("Upload a CSV to begin.")
    st.stop()

try:
    df = load_csv(uploaded_file.getvalue())
except Exception as exc:
    st.error(f"Could not read the file: {exc}")
    st.stop()

schema = infer_schema(df)
profile = basic_profile(df)
health_df = dataset_health_report(df)
recommendations = auto_module_recommendations(df, schema)
target_df = target_recommendations(df, schema)
rec_target = target_df.iloc[0]["column"] if not target_df.empty else None
rec_text = recommend_text_column(df, schema)
rec_date, rec_value = recommend_date_value_columns(schema)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{profile['shape'][0]:,}")
c2.metric("Columns", profile["shape"][1])
c3.metric("Duplicate rows", profile["duplicate_rows"])
c4.metric("Avg missing %", f"{df.isna().mean().mean() * 100:.1f}")

st.markdown('<div class="section-note"><strong>What betterlytics sees:</strong> It found your numeric, text, date, and target-like columns. It will now steer you toward the highest-value analyses instead of dumping every possible test on screen.</div>', unsafe_allow_html=True)

with st.expander("Detected data structure and beginner recommendations", expanded=True):
    left, right = st.columns(2)
    with left:
        st.write("**Detected columns by type**")
        st.json({
            "numeric": schema.numeric,
            "categorical": schema.categorical,
            "text": schema.text,
            "datetime": schema.datetime,
            "target_candidates": schema.target_candidates,
        })
    with right:
        st.write("**Recommended next choices**")
        st.write(f"Recommended target column: **{rec_target or 'None found yet'}**")
        st.write(f"Recommended text column: **{rec_text or 'None found yet'}**")
        st.write(f"Recommended date/value pair: **{rec_date or 'None'} / {rec_value or 'None'}**")
        st.write("Recommended modules:")
        st.json(recommendations)

with st.expander("Beginner controls", expanded=True):
    selected_target = st.selectbox(
        "Target column for prediction",
        options=[None] + schema.target_candidates,
        index=(1 if rec_target is not None else 0),
        help="Pick the outcome you want to predict. If you leave the recommended choice, betterlytics will use the strongest target candidate it found.",
    )
    st.caption("If you are not sure, leave the recommended choice in place and use the total scan.")

run_scan = st.button("Run total analytical scan", type="primary", help="This runs the strongest recommended analyses, fills a single dashboard, and writes a plain-English interpretation below it.")

if run_scan:
    scan = run_total_scan(df, schema, auto_optimize=auto_optimize, target_override=selected_target, test_size=test_size)
    scan["decision_engine"] = generate_decision_engine(scan)
    st.session_state["scan"] = scan
    st.success("Total analytical scan completed. The dashboard and plain-English interpretation are ready below.")

if "scan" not in st.session_state:
    st.info("Click 'Run total analytical scan' to populate the dashboard and the explanation section.")
    st.stop()

scan = st.session_state["scan"]

st.header("Executive dashboard")
dash1, dash2, dash3, dash4 = st.columns(4)
modeling = scan.get("modeling")
clustering = scan.get("clustering")
anomaly = scan.get("anomaly")
strong_corr = scan.get("correlations")

dash1.metric("Top target", modeling.get("target") if modeling else (rec_target or "None"))
if modeling and not modeling["leaderboard"].empty:
    lead = modeling["leaderboard"].iloc[0]
    metric_name = "Best accuracy" if modeling["problem_type"] == "classification" else "Best R2"
    metric_value = f"{lead['Accuracy']:.2f}" if modeling["problem_type"] == "classification" else f"{lead['R2']:.2f}"
    dash2.metric(metric_name, metric_value)
else:
    dash2.metric("Best model", "Not run")

dash3.metric("Segments found", clustering["k"] if clustering else "N/A")
dash4.metric("Anomalies flagged", anomaly["anomaly_count"] if anomaly else "N/A")

summary_tab, model_tab, segment_tab, text_tab, ts_tab, decision_tab, runlog_tab = st.tabs([
    "Overview",
    "Predictive modeling",
    "Segmentation & anomalies",
    "Text patterns",
    "Time trends",
    "Decision engine",
    "Run feedback",
])

with summary_tab:
    st.subheader("Top signals worth paying attention to")
    left, right = st.columns(2)
    with left:
        st.write("**Strongest correlations**")
        if strong_corr is not None and not strong_corr.empty:
            st.dataframe(strong_corr, use_container_width=True)
        else:
            st.info("No strong correlation table available yet because the dataset needs at least two numeric columns.")
        st.pyplot(make_missing_plot(df))
    with right:
        st.write("**Data health table**")
        st.dataframe(health_df.head(20), use_container_width=True)
        st.pyplot(make_corr_plot(df))

with model_tab:
    st.subheader("Predictive modeling leaderboard")
    st.caption("This section compares starter models and pushes the strongest performer to the top.")
    if modeling and not modeling["leaderboard"].empty:
        st.dataframe(modeling["leaderboard"].drop(columns=[c for c in ["report", "confusion_matrix", "score_for_ranking"] if c in modeling["leaderboard"].columns]), use_container_width=True)
        best_name = modeling["best_model"]
        st.write(f"**Best model:** {best_name}")
        if modeling["problem_type"] == "classification":
            report = modeling["leaderboard"].set_index("model").loc[best_name].get("report")
            cm = modeling["leaderboard"].set_index("model").loc[best_name].get("confusion_matrix")
            if isinstance(report, dict):
                st.write("Classification report")
                st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
            if cm is not None and not isinstance(cm, float):
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.imshow(cm)
                ax.set_title("Confusion matrix")
                fig.tight_layout()
                st.pyplot(fig)
    else:
        st.info("No predictive model ran. This usually means the app could not find a suitable target column.")

with segment_tab:
    st.subheader("Segmentation and unusual-record detection")
    a, b = st.columns(2)
    with a:
        st.write("**Clustering**")
        if clustering:
            st.metric("Best cluster count", clustering["k"])
            st.metric("Cluster quality", f"{clustering['silhouette']:.2f}")
            st.pyplot(make_cluster_plot(clustering["pca_df"]))
            st.dataframe(clustering["clustered_df"].head(25), use_container_width=True)
        else:
            st.info("Clustering was skipped because the dataset did not have enough numeric structure.")
    with b:
        st.write("**Anomaly detection**")
        if anomaly:
            st.metric("Potential anomalies", anomaly["anomaly_count"])
            st.pyplot(make_cluster_plot(anomaly["pca_df"]))
            st.dataframe(anomaly["scored_df"].head(25), use_container_width=True)
        else:
            st.info("Anomaly detection was skipped because the dataset did not have enough numeric structure.")

with text_tab:
    st.subheader("Text analytics")
    if scan.get("text"):
        st.write(f"Scanned text column: **{scan.get('text_col')}**")
        left, right = st.columns(2)
        with left:
            st.write("Top repeated terms")
            st.dataframe(scan["text"]["top_terms"], use_container_width=True)
        with right:
            st.write("Sentiment mix")
            st.dataframe(scan["text"]["sentiment"], use_container_width=True)
    else:
        st.info("No useful text-heavy column was detected for text analytics.")

with ts_tab:
    st.subheader("Time trend analysis")
    if scan.get("time_series"):
        st.write(f"Using **{scan.get('date_col')}** as the timeline and **{scan.get('value_col')}** as the tracked metric.")
        st.pyplot(make_time_plot(scan["time_series"]["series"], scan.get("date_col"), scan.get("value_col")))
        st.metric("Recent trend", scan["time_series"]["trend"])
    else:
        st.info("No usable date/value combination was found for time-series analysis.")


with decision_tab:
    st.subheader("Decision engine")
    de = scan.get("decision_engine")

    if not de:
        st.info("Decision engine has not been generated yet.")
    else:
        st.write("**Ranked actions**")
        st.dataframe(de["decision_df"], use_container_width=True)

        st.write("**Decision summary**")
        st.write(de["summary"])

        top_drivers = de.get("drivers", pd.DataFrame())
        if not top_drivers.empty:
            st.write("**Top model drivers**")
            st.dataframe(top_drivers, use_container_width=True)

        cluster_profiles = de.get("cluster_profiles", pd.DataFrame())
        if not cluster_profiles.empty:
            st.write("**Cluster profiles**")
            st.dataframe(cluster_profiles, use_container_width=True)

        left, right = st.columns(2)
        with left:
            if de.get("opportunities"):
                st.write("**Opportunities**")
                for item in de["opportunities"]:
                    st.write(f"- {item}")
        with right:
            if de.get("risks"):
                st.write("**Risks**")
                for item in de["risks"]:
                    st.write(f"- {item}")

        st.divider()
        st.subheader("Scenario lab")
        modeling = scan.get("modeling")
        if modeling and modeling.get("best_model"):
            target_name = modeling.get("target")
            editable_cols = [c for c in df.columns if c != target_name]
            numeric_edit_cols = [c for c in editable_cols if pd.api.types.is_numeric_dtype(df[c])]

            if not numeric_edit_cols:
                st.info("Scenario simulation needs at least one editable numeric feature.")
            else:
                sample_idx = st.number_input(
                    "Row index to simulate",
                    min_value=0,
                    max_value=max(0, len(df) - 1),
                    value=0,
                    step=1,
                )
                base_row = df.iloc[int(sample_idx)].copy()
                st.caption("Select a real row, then override a few numeric fields to see how the predicted outcome changes.")

                chosen_adjust_cols = st.multiselect(
                    "Numeric fields to modify",
                    options=numeric_edit_cols,
                    default=numeric_edit_cols[: min(3, len(numeric_edit_cols))],
                )

                adjustments = {}
                editor_cols = st.columns(min(3, max(1, len(chosen_adjust_cols)))) if chosen_adjust_cols else []
                for idx, col in enumerate(chosen_adjust_cols):
                    current_value = float(base_row[col]) if pd.notna(base_row[col]) else 0.0
                    with editor_cols[idx % len(editor_cols)]:
                        adjustments[col] = st.number_input(
                            f"Set {col}",
                            value=current_value,
                            key=f"scenario_{col}",
                        )

                if st.button("Run scenario simulation"):
                    result = scenario_simulator(scan, base_row, adjustments)
                    if result:
                        r1, r2, r3 = st.columns(3)
                        r1.metric("Original prediction", f"{result['original_prediction']}")
                        r2.metric("Adjusted prediction", f"{result['adjusted_prediction']}")
                        if result.get("delta") is not None:
                            r3.metric("Delta", f"{result['delta']:+.4f}")
                        else:
                            r3.metric("Delta", "N/A")
                        st.json(result)
                    else:
                        st.warning("Could not run the scenario simulation for this row and model.")
        else:
            st.info("Scenario simulation becomes available after a predictive model runs successfully.")

with runlog_tab:
    st.subheader("What the app did")
    st.caption("Every action gives feedback so a beginner always knows what ran, what was skipped, and why.")
    st.dataframe(pd.DataFrame(scan["run_log"]), use_container_width=True)

st.header("What the data means")
st.write(scan["narrative"])

st.subheader("Business implications")
for item in scan["implications"]:
    st.write(f"- {item}")

st.subheader("Recommended next actions")
actions = []
if modeling and not modeling["leaderboard"].empty:
    actions.append("Validate the best model on a fresh dataset or later time period before using it for decisions.")
if strong_corr is not None and not strong_corr.empty:
    actions.append("Investigate whether the strongest relationships are causal, coincidental, or driven by data leakage.")
if clustering:
    actions.append("Name the discovered segments in business terms and test whether they respond differently to marketing actions.")
if anomaly:
    actions.append("Review the flagged anomalies manually to decide whether they are errors, edge cases, or high-value exceptions.")
if not actions:
    actions.append("Start by improving data quality and collecting clearer outcome variables, then rerun the total scan.")
for item in actions:
    st.write(f"- {item}")

st.header("Downloads")
st.download_button("Download cleaned CSV", data=df.to_csv(index=False).encode("utf-8"), file_name=f"betterlytics_cleaned_{uploaded_file.name}", mime="text/csv")
st.download_button("Download interpretation summary", data=scan["narrative"].encode("utf-8"), file_name="betterlytics_interpretation.txt", mime="text/plain")
st.download_button("Download scan data as JSON", data=safe_json(scan).encode("utf-8"), file_name="betterlytics_scan.json", mime="application/json")

add app files


