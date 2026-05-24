"""
Streamlit audit dashboard for the Loan Decision AI Audit Framework.

Launch with:
    streamlit run src/dashboard/app.py

Three tabs:
    1. Decision Explorer — pick an applicant, see prediction + SHAP reason codes
    2. Drift Monitor     — PSI summary + performance trend from Step 4
    3. Model Versions    — MLflow runs with audit / monitor / feedback tags
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make `src/...` importable when launched via `streamlit run src/dashboard/app.py`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import mlflow
import shap
import xgboost as xgb

from src.utils.config import get_paths, load_config
from src.utils.mlflow_helpers import configure_mlflow
from src.data.loader import load_engineered_train
from src.audit.shap_explain import (
    per_applicant_reason_codes,
    plot_shap_waterfall,
)


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Loan Decision Audit",
    page_icon="🏦",
)


# ─────────────────────────────────────────────────────────────────────────────
# Cached loaders. Streamlit re-runs the whole script on every interaction;
# without caching, we'd re-load 34 MB of parquet and re-score 307K applicants
# every time someone clicks a dropdown. `cache_resource` keeps Python objects
# (models, explainers) alive; `cache_data` memoises return values keyed by
# args.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model bundle from MLflow…")
def load_model_bundle():
    """Load audited model + preprocessor + feature names. Returns a dict."""
    cfg = load_config()
    paths = get_paths(cfg)
    configure_mlflow(experiment="loan-default-baseline")

    runs = mlflow.search_runs(
        experiment_names=["loan-default-baseline"],
        filter_string='attributes.status = "FINISHED" and tags.audited = "true"',
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        st.error("No audited model found in MLflow — complete Steps 2-3 first.")
        st.stop()

    run_id = runs.iloc[0]["run_id"]
    model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
    preprocessor = joblib.load(paths.artifacts / "preprocessor.joblib")
    with open(paths.artifacts / "feature_names.json") as fh:
        feature_names = json.load(fh)

    return {
        "cfg": cfg,
        "paths": paths,
        "run_id": run_id,
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
    }


@st.cache_data(show_spinner="Scoring all applicants…")
def load_applicants_with_scores():
    """Load engineered applicants, score them with the baseline model."""
    bundle = load_model_bundle()
    df = load_engineered_train()
    cfg = bundle["cfg"]

    id_col = cfg["project"]["id_col"]
    target_col = cfg["project"]["target_col"]

    X = df.drop(columns=[target_col, id_col])
    X_mat = bundle["preprocessor"].transform(X)
    X_df = pd.DataFrame(X_mat, columns=bundle["feature_names"])
    scores = bundle["model"].predict(xgb.DMatrix(X_df))
    df = df.assign(_score=scores).reset_index(drop=True)
    return df, X_mat


@st.cache_resource(show_spinner="Building SHAP explainer…")
def get_explainer():
    """Cache the TreeExplainer — building it is fast; reusing it is faster."""
    bundle = load_model_bundle()
    return shap.TreeExplainer(bundle["model"])


@st.cache_data
def load_artifact_csv(filename: str):
    bundle = load_model_bundle()
    path = bundle["paths"].artifacts / filename
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def list_all_runs():
    configure_mlflow(experiment="loan-default-baseline")
    return mlflow.search_runs(
        experiment_names=["loan-default-baseline"],
        order_by=["attributes.start_time DESC"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Load everything once at top of script (cached, so this is fast on rerun)
# ─────────────────────────────────────────────────────────────────────────────
bundle = load_model_bundle()
df, X_mat = load_applicants_with_scores()
explainer = get_explainer()

ID_COL = bundle["cfg"]["project"]["id_col"]
TARGET = bundle["cfg"]["project"]["target_col"]


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("🏦 Loan Decision AI Audit & Reliability Framework")
st.caption(
    f"Auditing MLflow run **`{bundle['run_id']}`**  ·  "
    f"{len(df):,} applicants scored  ·  "
    f"baseline default rate: **{df[TARGET].mean():.2%}**"
)


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_decision, tab_drift, tab_versions = st.tabs([
    "🔍 Decision Explorer",
    "📈 Drift Monitor",
    "📦 Model Versions",
])


# ── Tab 1 ────────────────────────────────────────────────────────────────────
with tab_decision:
    st.header("Per-applicant decision audit")
    st.markdown(
        "Pick any applicant by ID to see the model's predicted probability, "
        "the SHAP reason codes that drove it, and the applicant's input snapshot."
    )

    col_filter, col_pick = st.columns([1, 2])

    with col_filter:
        bucket = st.radio(
            "Quick filter",
            options=[
                "Any",
                "High-risk (score ≥ 0.30)",
                "Borderline (0.10 ≤ score ≤ 0.20)",
                "Low-risk (score ≤ 0.05)",
            ],
            index=0,
        )
        if bucket == "High-risk (score ≥ 0.30)":
            filtered = df[df["_score"] >= 0.30]
        elif bucket == "Borderline (0.10 ≤ score ≤ 0.20)":
            filtered = df[(df["_score"] >= 0.10) & (df["_score"] <= 0.20)]
        elif bucket == "Low-risk (score ≤ 0.05)":
            filtered = df[df["_score"] <= 0.05]
        else:
            filtered = df

        st.caption(f"{len(filtered):,} applicants match")

    with col_pick:
        # Cap the dropdown at 500 options for snappy interaction. The picker is
        # for demonstration, not exhaustive review.
        id_options = filtered[ID_COL].head(500).tolist()
        if not id_options:
            st.warning("No applicants match this filter.")
            st.stop()
        applicant_id = st.selectbox(
            "Applicant ID (SK_ID_CURR)",
            options=id_options,
            index=0,
        )

    # Headline metrics for the selected applicant
    row = df[df[ID_COL] == applicant_id].iloc[0]
    row_idx = int(df.index[df[ID_COL] == applicant_id][0])
    score = float(row["_score"])
    actual = int(row[TARGET])
    DEMO_THRESHOLD = 0.15  # Roughly matches the Step 3 operating threshold

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model score", f"{score:.4f}")
    m2.metric(
        "Decision",
        "REJECT" if score >= DEMO_THRESHOLD else "APPROVE",
        delta=f"threshold = {DEMO_THRESHOLD}",
        delta_color="off",
    )
    m3.metric("Actual outcome", "Defaulted" if actual == 1 else "Repaid")
    correct = (score >= DEMO_THRESHOLD) == (actual == 1)
    m4.metric("Prediction correct?", "Yes" if correct else "No")

    st.divider()

    col_waterfall, col_reasons = st.columns([3, 2])

    with col_waterfall:
        st.subheader("SHAP waterfall (per-decision attribution)")

        # Compute SHAP for this single applicant — fast because explainer is cached
        row_X = X_mat[row_idx : row_idx + 1]
        row_df = pd.DataFrame(row_X, columns=bundle["feature_names"])
        sv = explainer.shap_values(row_df)

        fig = plot_shap_waterfall(
            shap_values=sv,
            expected_value=float(explainer.expected_value),
            X=row_X,
            feature_names=bundle["feature_names"],
            idx=0,
            max_display=10,
        )
        st.pyplot(fig)
        plt.close(fig)

    with col_reasons:
        st.subheader("Top reason codes")
        reasons = per_applicant_reason_codes(sv, bundle["feature_names"], 0, top_n=5)
        st.dataframe(reasons, hide_index=True, use_container_width=True)

        st.subheader("Applicant inputs (snapshot)")
        display_cols = [
            "AGE_YEARS", "EMPLOYED_YEARS", "AMT_INCOME_TOTAL",
            "AMT_CREDIT", "AMT_ANNUITY", "CREDIT_INCOME_RATIO",
            "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
            "CODE_GENDER", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
        ]
        display_cols = [c for c in display_cols if c in df.columns]
        snapshot = pd.DataFrame({
            "feature": display_cols,
            "value": [row[c] for c in display_cols],
        })
        st.dataframe(snapshot, hide_index=True, use_container_width=True)


# ── Tab 2 ────────────────────────────────────────────────────────────────────
with tab_drift:
    st.header("Drift status")
    st.markdown(
        "PSI snapshot and performance trend from Step 4's synthetic six-month "
        "monitoring window. PSI is computed per feature against the training "
        "reference. Severity thresholds: stable < 0.10, moderate < 0.25, "
        "significant ≥ 0.25."
    )

    severity = load_artifact_csv("drift_severity_counts.csv")
    psi_matrix = load_artifact_csv("drift_psi_matrix.csv")
    perf = load_artifact_csv("drift_performance_by_month.csv")

    if severity is None:
        st.warning(
            "No drift artifacts found in `artifacts/`. Run Step 4's notebook "
            "to populate this tab."
        )
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Features per severity bucket by month")
            st.dataframe(severity, use_container_width=True)

        with c2:
            if perf is not None:
                st.subheader("Headline metrics")
                m1, m2, m3 = st.columns(3)
                m1.metric("Month-1 PR-AUC", f"{perf.iloc[0]['pr_auc']:.4f}")
                m2.metric("Month-6 PR-AUC", f"{perf.iloc[-1]['pr_auc']:.4f}",
                          delta=f"{perf.iloc[-1]['pr_auc'] - perf.iloc[0]['pr_auc']:+.4f}")
                m3.metric("Month-6 base rate", f"{perf.iloc[-1]['base_rate']:.2%}")

        if psi_matrix is not None:
            st.subheader("Top 15 most-drifted features (PSI by month)")
            first_col = psi_matrix.columns[0]
            psi_top = psi_matrix.head(15).set_index(first_col)
            st.dataframe(psi_top.round(3), use_container_width=True)

        if perf is not None:
            st.subheader("Performance trajectory")
            fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
            axes[0].plot(perf["month"], perf["pr_auc"], marker="o", lw=2, color="#4C72B0")
            axes[0].set_title("PR-AUC over six months")
            axes[0].set_xlabel("month"); axes[0].set_ylabel("PR-AUC"); axes[0].grid(alpha=0.3)
            axes[1].plot(perf["month"], perf["mean_score"], marker="o", lw=2,
                         color="#C44E52", label="mean score")
            axes[1].plot(perf["month"], perf["base_rate"], marker="s", lw=2,
                         color="black", label="actual base rate")
            axes[1].set_title("Mean score vs actual base rate")
            axes[1].set_xlabel("month"); axes[1].set_ylabel("rate")
            axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
            for ax in axes:
                for spine in ("top", "right"):
                    ax.spines[spine].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


# ── Tab 3 ────────────────────────────────────────────────────────────────────
with tab_versions:
    st.header("Model version history")
    st.markdown(
        "Every MLflow run in the experiment, newest first. Tags surface which "
        "runs were audited, monitored, and had feedback loops applied. "
        "Nested child runs are the corrected versions from Step 5."
    )

    runs_df = list_all_runs()

    # Friendly columns — drop the noisy `metrics.` / `tags.` / `params.` prefixes.
    keep_cols = [
        "run_id",
        "tags.mlflow.runName",
        "status",
        "start_time",
        "metrics.pr_auc",
        "metrics.recall_at_p50",
        "tags.audited",
        "tags.monitored",
        "tags.role",
        "tags.correction_batch",
        "params.best_iteration",
    ]
    keep_cols = [c for c in keep_cols if c in runs_df.columns]
    pretty = runs_df[keep_cols].copy()
    pretty.columns = [
        c.replace("metrics.", "").replace("tags.", "").replace("params.", "")
         .replace("mlflow.", "")
        for c in pretty.columns
    ]
    st.dataframe(pretty, hide_index=True, use_container_width=True)

    st.subheader("Headline audit + monitoring + feedback metrics (latest run)")
    summary_cols = [
        c for c in runs_df.columns
        if c.startswith(("metrics.audit_", "metrics.drift_", "metrics.feedback_"))
    ]
    if summary_cols and not runs_df.empty:
        latest = runs_df.iloc[0][summary_cols].dropna()
        if not latest.empty:
            summary_df = pd.DataFrame({
                "metric": [c.replace("metrics.", "") for c in latest.index],
                "value": [round(float(v), 4) for v in latest.values],
            })
            st.dataframe(summary_df, hide_index=True, use_container_width=True)
        else:
            st.caption("No audit / drift / feedback metrics on the most recent run.")


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Loan Decision AI Audit & Reliability Framework  ·  "
    "Built on XGBoost · SHAP · Evidently · MLflow · Streamlit  ·  "
    "Home Credit Default Risk dataset"
)
