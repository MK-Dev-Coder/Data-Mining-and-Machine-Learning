"""
Step 3: identifying unengaged / churned users.

Two approaches are produced:

1. Heuristic baseline: a user is churned if their last transaction is older
   than the lookback window. Useful as a baseline and as a no-ML option the
   business could deploy directly.

2. Machine-learning models: Logistic Regression and Random Forest fit on the
   engineered features (no leakage). Both are evaluated on a stratified 20%
   test set with ROC-AUC, PR-AUC, F1 and a confusion matrix.

All artefacts (metric tables, plots, fitted model) are written to outputs/.
"""
from __future__ import annotations
import json
import logging
from typing import Dict

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config

log = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="notebook")


# ---------------------------------------------------------------------------
# Heuristic baseline
# ---------------------------------------------------------------------------
def heuristic_classify(meta: pd.DataFrame) -> pd.Series:
    """A user is churned if they had no transaction for >= LOOKBACK_DAYS."""
    return (meta["days_since_last_trx"] >= config.LOOKBACK_DAYS).astype(int)


# ---------------------------------------------------------------------------
# ML models
# ---------------------------------------------------------------------------
def _build_pipelines() -> Dict[str, Pipeline]:
    numeric_pre = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True)),
        ]
    )
    # Trees don't need scaling, so we skip the StandardScaler for RF.
    numeric_pre_rf = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    return {
        "logreg": Pipeline(
            steps=[
                ("pre", numeric_pre),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=config.RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "rf": Pipeline(
            steps=[
                ("pre", numeric_pre_rf),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_leaf=20,
                        n_jobs=-1,
                        class_weight="balanced",
                        random_state=config.RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def _evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
    }


def _plot_roc_pr(results: dict, y_test: np.ndarray) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["proba"])
        ax[0].plot(fpr, tpr, label=f"{name} (AUC={r['metrics']['roc_auc']:.3f})")
        prec, rec, _ = precision_recall_curve(y_test, r["proba"])
        ax[1].plot(rec, prec, label=f"{name} (AP={r['metrics']['pr_auc']:.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].set_title("ROC")
    ax[0].legend(loc="lower right")
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("Precision-Recall")
    ax[1].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "model_roc_pr.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["engaged", "churned"],
        yticklabels=["engaged", "churned"], ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(f"Confusion matrix - {name}")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"confusion_{name}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_importance(model: Pipeline, feature_names: list[str], top: int = 25) -> None:
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
        title = "Random Forest - feature importances"
        out = "feature_importances_rf.png"
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_[0])
        title = "Logistic Regression - |coef|"
        out = "feature_importances_logreg.png"
    else:
        return
    s = pd.Series(imp, index=feature_names).sort_values(ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(8, max(4, top * 0.25)))
    s.plot.barh(ax=ax, color="steelblue")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / out, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_modelling(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame) -> pd.DataFrame:
    metrics_rows: list[dict] = []

    # --- heuristic baseline (evaluated on the entire eligible population) -
    h_pred = heuristic_classify(meta).values
    h_report = classification_report(y, h_pred, output_dict=True, zero_division=0)
    metrics_rows.append(
        {
            "model": "heuristic_recency",
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "f1": f1_score(y, h_pred, zero_division=0),
            "precision": h_report["1"]["precision"],
            "recall": h_report["1"]["recall"],
        }
    )
    log.info(
        "Heuristic recency baseline: F1=%.3f  P=%.3f  R=%.3f",
        metrics_rows[-1]["f1"], metrics_rows[-1]["precision"], metrics_rows[-1]["recall"],
    )

    # --- train / test split -----------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE
    )

    pipes = _build_pipelines()
    results: Dict[str, dict] = {}
    best_name, best_auc = None, -1.0

    for name, pipe in pipes.items():
        log.info("Training %s ...", name)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        m = _evaluate(name, y_test.values, pred, proba)
        log.info("  %s: ROC-AUC=%.3f  PR-AUC=%.3f  F1=%.3f", name, m["roc_auc"], m["pr_auc"], m["f1"])
        metrics_rows.append(m)
        results[name] = {"pipe": pipe, "proba": proba, "pred": pred, "metrics": m}

        _plot_confusion(name, y_test.values, pred)
        _plot_feature_importance(pipe, list(X.columns))

        if m["roc_auc"] > best_auc:
            best_auc, best_name = m["roc_auc"], name

    _plot_roc_pr(results, y_test.values)

    # save metrics
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(config.TABLES_DIR / "model_metrics.csv", index=False)
    log.info("\n%s", metrics_df.to_string(index=False))

    # save best model + scoring summary
    best = results[best_name]
    joblib.dump(best["pipe"], config.MODELS_DIR / f"best_model_{best_name}.joblib")

    # --- score the entire eligible population for downstream business use --
    full_proba = best["pipe"].predict_proba(X)[:, 1]
    full_pred = (full_proba >= 0.5).astype(int)
    predictions = meta[["user_id"]].copy()
    predictions["true_churned"] = y.values
    predictions["pred_churn_proba"] = full_proba
    predictions["pred_churned"] = full_pred
    predictions["heuristic_churned"] = h_pred
    predictions.to_csv(config.TABLES_DIR / "predictions.csv", index=False)

    n_pred_churn_model = int(full_pred.sum())
    n_pred_churn_heur = int(h_pred.sum())
    log.info(
        "Population-level churn flags: model=%d (%.1f%%), heuristic=%d (%.1f%%), actual=%d (%.1f%%)",
        n_pred_churn_model, 100 * n_pred_churn_model / len(X),
        n_pred_churn_heur, 100 * n_pred_churn_heur / len(X),
        int(y.sum()), 100 * y.mean(),
    )

    summary = {
        "snapshot": str(meta["snapshot"].iloc[0].date()),
        "cutoff": str(meta["cutoff"].iloc[0].date()),
        "lookback_days": config.LOOKBACK_DAYS,
        "n_users_modelled": int(len(meta)),
        "n_churned_actual": int(y.sum()),
        "churn_rate_actual": float(y.mean()),
        "n_churn_predicted_model": n_pred_churn_model,
        "n_churn_predicted_heuristic": n_pred_churn_heur,
        "best_model": best_name,
        "best_metrics": best["metrics"],
    }
    with open(config.TABLES_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("Best model: %s (saved). Summary written to %s",
             best_name, config.TABLES_DIR / "summary.json")
    return metrics_df
