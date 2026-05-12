"""
Streamlit dashboard for the CCS6521 churn-prediction project.

Run with:

    streamlit run app.py

The dashboard reads pre-computed artefacts from outputs/ (figures, tables and
the pickled model). If those don't exist yet, run `python -m src.main` first.
"""
from __future__ import annotations
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
MOD = ROOT / "outputs" / "models"

st.set_page_config(
    page_title="CCS6521 - Churn Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_summary() -> dict:
    import json
    p = TAB / "summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data
def load_metrics() -> pd.DataFrame:
    p = TAB / "model_metrics.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_predictions() -> pd.DataFrame:
    p = TAB / "predictions.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_meta() -> pd.DataFrame:
    p = TAB / "meta.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_features() -> pd.DataFrame:
    """Loads the feature matrix. Not cached so newly-pushed files are picked up."""
    p = TAB / "feature_matrix.csv"
    if not p.exists():
        st.error(f"Feature matrix not found at: `{p}`")
        try:
            siblings = sorted(TAB.iterdir())
            st.write("Files actually present in `outputs/tables/`:")
            st.write([s.name for s in siblings])
        except Exception as e:
            st.write(f"(could not list outputs/tables/: {e})")
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception as e:
        st.error(f"Failed to read {p}: {e}")
        return pd.DataFrame()


@st.cache_resource
def load_model():
    p = MOD / "best_model_rf.joblib"
    return joblib.load(p) if p.exists() else None


def show_image(name: str, caption: str | None = None) -> None:
    p = FIG / name
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing figure: {name}. Run `python -m src.main` first.")


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
PAGES = [
    "Overview",
    "EDA - Users & Devices",
    "EDA - Transactions",
    "EDA - Notifications",
    "EDA - Relationships",
    "Model results",
    "Predictions explorer",
    "Score a user",
]
page = st.sidebar.radio("Section", PAGES)
st.sidebar.caption(
    "Data is pre-computed in `outputs/`. Re-run "
    "`python -m src.main` to refresh after changing the pipeline."
)

summary = load_summary()
metrics = load_metrics()
preds = load_predictions()
meta = load_meta()


# ---------------------------------------------------------------------------
# 1. Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Churn-prediction dashboard")
    st.markdown(
        "FinTech churn-prediction Knowledge-Discovery pipeline. "
        "EDA, preprocessing, engagement labelling and modelling, all from four raw CSV tables."
    )
    if summary:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Eligible users", f"{summary['n_users_modelled']:,}")
        c2.metric("Actual churn", f"{summary['n_churned_actual']:,}",
                  f"{100*summary['churn_rate_actual']:.1f}%")
        c3.metric("Predicted (model)", f"{summary['n_churn_predicted_model']:,}")
        c4.metric("Predicted (heuristic)", f"{summary['n_churn_predicted_heuristic']:,}")

        st.markdown(
            f"**Snapshot:** {summary['snapshot']} &nbsp;&nbsp; "
            f"**Cutoff:** {summary['cutoff']} &nbsp;&nbsp; "
            f"**Lookback:** {summary['lookback_days']} days &nbsp;&nbsp; "
            f"**Best model:** `{summary['best_model']}`"
        )

    st.subheader("Headline metrics")
    if not metrics.empty:
        st.dataframe(metrics.style.format({
            "roc_auc": "{:.3f}", "pr_auc": "{:.3f}",
            "f1": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}",
        }), use_container_width=True)
    else:
        st.info("Run `python -m src.main` to generate metrics.")

    st.subheader("ROC and Precision-Recall")
    show_image("model_roc_pr.png")


# ---------------------------------------------------------------------------
# 2. EDA - Users & Devices
# ---------------------------------------------------------------------------
elif page == "EDA - Users & Devices":
    st.title("Users & Devices")
    st.caption(
        "Demographic and account-level distributions for the 15,544-user population."
    )
    show_image("users_overview.png", "Birth year, plan, top-15 countries, signups over time.")
    col1, col2 = st.columns(2)
    with col1:
        show_image("users_contacts_referrals.png", "log(1 + num_contacts) and log(1 + num_referrals).")
    with col2:
        show_image("devices_overview.png", "Device brand counts and brand share by plan.")


# ---------------------------------------------------------------------------
# 3. EDA - Transactions
# ---------------------------------------------------------------------------
elif page == "EDA - Transactions":
    st.title("Transactions")
    st.caption("2.18 M transactions, Jan 2018 to mid-May 2019.")
    show_image("transactions_overview.png", "Type, state, log-amount, weekly volume.")
    show_image("transactions_currencies_countries.png", "Top-15 currencies and merchant countries.")


# ---------------------------------------------------------------------------
# 4. EDA - Notifications
# ---------------------------------------------------------------------------
elif page == "EDA - Notifications":
    st.title("Notifications")
    st.caption("97,704 outbound notifications across EMAIL, PUSH and SMS.")
    show_image("notifications_overview.png", "Channel, status, top reasons, weekly volume.")


# ---------------------------------------------------------------------------
# 5. EDA - Relationships
# ---------------------------------------------------------------------------
elif page == "EDA - Relationships":
    st.title("Cross-table relationships")
    col1, col2 = st.columns(2)
    with col1:
        show_image("correlation_heatmap.png",
                   "Correlation of log-scaled volume features.")
    with col2:
        show_image("activity_by_plan.png",
                   "Transaction volume and value by plan tier.")


# ---------------------------------------------------------------------------
# 6. Model results
# ---------------------------------------------------------------------------
elif page == "Model results":
    st.title("Model evaluation")

    if not metrics.empty:
        st.subheader("Metrics on stratified 20% test set")
        st.dataframe(metrics.style.format({
            "roc_auc": "{:.3f}", "pr_auc": "{:.3f}",
            "f1": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}",
        }), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        show_image("model_roc_pr.png", "ROC and Precision-Recall.")
        show_image("confusion_logreg.png", "Logistic Regression - confusion matrix.")
    with col2:
        show_image("feature_importances_rf.png",
                   "Random Forest feature importances (top 25).")
        show_image("confusion_rf.png", "Random Forest - confusion matrix.")
    show_image("feature_importances_logreg.png",
               "Logistic Regression - |coefficients| (top 25).")


# ---------------------------------------------------------------------------
# 7. Predictions explorer
# ---------------------------------------------------------------------------
elif page == "Predictions explorer":
    st.title("Predictions explorer")
    if preds.empty:
        st.info("Run `python -m src.main` to generate predictions.")
    else:
        threshold = st.slider(
            "Probability threshold for 'churned'",
            min_value=0.0, max_value=1.0, value=0.50, step=0.01,
        )

        df = preds.copy()
        df["pred_at_threshold"] = (df["pred_churn_proba"] >= threshold).astype(int)

        flagged = int(df["pred_at_threshold"].sum())
        actual = int(df["true_churned"].sum())
        tp = int(((df["pred_at_threshold"] == 1) & (df["true_churned"] == 1)).sum())
        fp = int(((df["pred_at_threshold"] == 1) & (df["true_churned"] == 0)).sum())
        fn = int(((df["pred_at_threshold"] == 0) & (df["true_churned"] == 1)).sum())
        tn = int(((df["pred_at_threshold"] == 0) & (df["true_churned"] == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Flagged", f"{flagged:,}", f"{100*flagged/len(df):.1f}%")
        c2.metric("Precision", f"{precision:.3f}")
        c3.metric("Recall", f"{recall:.3f}")
        c4.metric("F1", f"{f1:.3f}")

        st.markdown(
            f"**True positives:** {tp:,} &nbsp;&nbsp; "
            f"**False positives:** {fp:,} &nbsp;&nbsp; "
            f"**False negatives:** {fn:,} &nbsp;&nbsp; "
            f"**True negatives:** {tn:,}"
        )

        st.subheader("Score distribution")
        st.bar_chart(
            df["pred_churn_proba"].round(2).value_counts().sort_index()
        )

        st.subheader("Filtered predictions")
        only_flagged = st.checkbox("Show only flagged users", value=False)
        only_misclassified = st.checkbox("Show only misclassified users", value=False)
        view = df.copy()
        if only_flagged:
            view = view[view["pred_at_threshold"] == 1]
        if only_misclassified:
            view = view[view["pred_at_threshold"] != view["true_churned"]]

        view = view.sort_values("pred_churn_proba", ascending=False).head(2000)
        st.dataframe(view, use_container_width=True, height=420)
        st.caption(f"Showing first {min(len(view), 2000):,} rows after filters.")

        st.download_button(
            "Download flagged users (CSV)",
            data=df[df["pred_at_threshold"] == 1].to_csv(index=False).encode(),
            file_name=f"flagged_users_threshold_{threshold:.2f}.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# 8. Score a user
# ---------------------------------------------------------------------------
elif page == "Score a user":
    st.title("Score a user")
    st.caption(
        "Pick any user from the population and see the model's predicted "
        "churn probability, the actual outcome and the most influential features."
    )

    if preds.empty or load_features().empty:
        st.info("Run `python -m src.main` first.")
    else:
        features = load_features()
        users_list = preds["user_id"].sort_values().tolist()

        col_a, col_b = st.columns([1, 2])
        with col_a:
            uid = st.selectbox("user_id", users_list, index=0)
        row = preds[preds["user_id"] == uid].iloc[0]
        with col_b:
            c1, c2, c3 = st.columns(3)
            c1.metric("Pred. churn probability", f"{row['pred_churn_proba']:.3f}")
            c2.metric("Predicted (>=0.5)", "churned" if row["pred_churned"] else "engaged")
            c3.metric("Actual", "churned" if row["true_churned"] else "engaged")

        # Show top features for this user (by importance)
        model = load_model()
        if model is not None:
            clf = model.named_steps.get("clf")
            try:
                imp = clf.feature_importances_
                idx = preds[preds["user_id"] == uid].index[0]
                user_row = features.iloc[idx]
                top = (
                    pd.Series(imp, index=features.columns)
                    .sort_values(ascending=False)
                    .head(10)
                )
                top_df = pd.DataFrame({
                    "feature": top.index,
                    "importance": top.values,
                    "this_user_value": user_row[top.index].values,
                })
                st.subheader("Top-10 features by global importance")
                st.dataframe(top_df, use_container_width=True)
            except AttributeError:
                pass
