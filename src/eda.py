"""
Step 1: Exploratory Data Analysis.

Generates descriptive-statistics tables and plots covering each of the four
tables and the most important relationships between them. Outputs are written
to outputs/figures and outputs/tables; nothing is printed except a short
summary log.
"""
from __future__ import annotations
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; we only save PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config

log = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="notebook")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save_fig(fig: plt.Figure, name: str) -> Path:
    out = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def _missing_summary(df: pd.DataFrame, name: str) -> pd.DataFrame:
    miss = df.isna().sum()
    pct = 100 * miss / len(df)
    out = pd.DataFrame({"n_missing": miss, "pct_missing": pct.round(2)})
    out = out[out.n_missing > 0].sort_values("pct_missing", ascending=False)
    out.to_csv(config.TABLES_DIR / f"missing_{name}.csv")
    return out


# ---------------------------------------------------------------------------
# Per-table EDA
# ---------------------------------------------------------------------------
def eda_users(users: pd.DataFrame) -> None:
    log.info("EDA: users (%d rows)", len(users))
    users.describe(include="all").to_csv(config.TABLES_DIR / "describe_users.csv")
    _missing_summary(users, "users")

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    sns.histplot(users["birth_year"], bins=30, kde=True, ax=ax[0, 0])
    ax[0, 0].set_title("User birth year")
    users["plan"].value_counts().plot.bar(ax=ax[0, 1], color="steelblue")
    ax[0, 1].set_title("Plan distribution")
    ax[0, 1].set_ylabel("users")
    users["country"].value_counts().head(15).plot.bar(ax=ax[1, 0], color="seagreen")
    ax[1, 0].set_title("Top-15 countries")
    ax[1, 0].set_ylabel("users")
    sns.histplot(users["created_date"], bins=40, ax=ax[1, 1])
    ax[1, 1].set_title("User signups over time")
    _save_fig(fig, "users_overview.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(np.log1p(users["num_contacts"]), bins=40, ax=ax[0])
    ax[0].set_title("log(1 + num_contacts)")
    sns.histplot(np.log1p(users["num_referrals"]), bins=40, ax=ax[1])
    ax[1].set_title("log(1 + num_referrals)")
    _save_fig(fig, "users_contacts_referrals.png")


def eda_devices(devices: pd.DataFrame, users: pd.DataFrame) -> None:
    log.info("EDA: devices (%d rows)", len(devices))
    counts = devices["brand"].value_counts()
    counts.to_csv(config.TABLES_DIR / "devices_brand_counts.csv", header=["count"])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    counts.plot.bar(ax=ax[0], color="indigo")
    ax[0].set_title("Device brand counts")

    merged = devices.merge(users[["user_id", "plan"]], on="user_id", how="left")
    pivot = (
        merged.groupby(["plan", "brand"]).size().unstack(fill_value=0)
    )
    pivot.div(pivot.sum(axis=1), axis=0).plot.bar(stacked=True, ax=ax[1])
    ax[1].set_title("Device brand share by plan")
    ax[1].set_ylabel("share")
    _save_fig(fig, "devices_overview.png")


def eda_notifications(notifs: pd.DataFrame) -> None:
    log.info("EDA: notifications (%d rows)", len(notifs))
    _missing_summary(notifs, "notifications")
    notifs["reason"].value_counts().to_csv(
        config.TABLES_DIR / "notifications_reason_counts.csv", header=["count"]
    )

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    notifs["channel"].value_counts().plot.bar(ax=ax[0, 0], color="orange")
    ax[0, 0].set_title("Channel")
    notifs["status"].value_counts().plot.bar(ax=ax[0, 1], color="firebrick")
    ax[0, 1].set_title("Status")
    notifs["reason"].value_counts().head(12).plot.barh(ax=ax[1, 0], color="teal")
    ax[1, 0].set_title("Top-12 reasons")
    notifs.set_index("created_date").resample("W").size().plot(ax=ax[1, 1])
    ax[1, 1].set_title("Notifications per week")
    ax[1, 1].set_ylabel("count")
    _save_fig(fig, "notifications_overview.png")


def eda_transactions(trx: pd.DataFrame) -> None:
    log.info("EDA: transactions (%d rows)", len(trx))
    _missing_summary(trx, "transactions")

    # Cap the absurd outliers when describing the amount
    trx_clean = trx.copy()
    trx_clean["amount_usd_log"] = np.log1p(trx_clean["amount_usd"].clip(lower=0))
    trx_clean[["amount_usd"]].describe().to_csv(
        config.TABLES_DIR / "describe_transactions_amount.csv"
    )

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    trx_clean["transactions_type"].value_counts().plot.bar(ax=ax[0, 0], color="steelblue")
    ax[0, 0].set_title("Transaction type")
    trx_clean["transactions_state"].value_counts().plot.bar(ax=ax[0, 1], color="darkgreen")
    ax[0, 1].set_title("Transaction state")
    sns.histplot(trx_clean["amount_usd_log"], bins=80, ax=ax[1, 0])
    ax[1, 0].set_title("log(1 + amount_usd)")
    trx_clean.set_index("created_date").resample("W").size().plot(ax=ax[1, 1])
    ax[1, 1].set_title("Transactions per week")
    ax[1, 1].set_ylabel("count")
    _save_fig(fig, "transactions_overview.png")

    # Top currencies & merchant countries
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    trx["transactions_currency"].value_counts().head(15).plot.bar(ax=ax[0], color="navy")
    ax[0].set_title("Top-15 currencies")
    trx["ea_merchant_country"].value_counts().head(15).plot.bar(ax=ax[1], color="darkred")
    ax[1].set_title("Top-15 merchant countries")
    _save_fig(fig, "transactions_currencies_countries.png")


# ---------------------------------------------------------------------------
# Cross-table relationships
# ---------------------------------------------------------------------------
def eda_relationships(
    users: pd.DataFrame,
    devices: pd.DataFrame,
    notifs: pd.DataFrame,
    trx: pd.DataFrame,
) -> None:
    log.info("EDA: cross-table relationships")

    user_trx_counts = trx.groupby("user_id").size().rename("n_transactions")
    user_trx_value = (
        trx[trx["transactions_state"] == "COMPLETED"]
        .groupby("user_id")["amount_usd"]
        .sum()
        .rename("total_completed_usd")
    )
    user_notif_counts = notifs.groupby("user_id").size().rename("n_notifications")

    enriched = (
        users.set_index("user_id")
        .join([user_trx_counts, user_trx_value, user_notif_counts], how="left")
        .fillna({"n_transactions": 0, "total_completed_usd": 0, "n_notifications": 0})
    )
    enriched["age"] = 2019 - enriched["birth_year"]
    enriched.to_csv(config.TABLES_DIR / "users_enriched_for_eda.csv")

    # Correlation heatmap on numeric, log-scaled where helpful
    num = enriched[
        [
            "age",
            "user_settings_crypto_unlocked",
            "num_contacts",
            "num_referrals",
            "num_successful_referrals",
            "n_transactions",
            "total_completed_usd",
            "n_notifications",
        ]
    ].copy()
    for c in ["num_contacts", "num_referrals", "n_transactions", "total_completed_usd",
              "n_notifications"]:
        num[c] = np.log1p(num[c].clip(lower=0))

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(num.corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
    ax.set_title("Correlation (log-scaled where appropriate)")
    _save_fig(fig, "correlation_heatmap.png")

    # Activity by plan
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(
        data=enriched.reset_index(),
        x="plan",
        y=np.log1p(enriched["n_transactions"]).values,
        ax=ax[0],
        order=["STANDARD", "SILVER", "GOLD"],
    )
    ax[0].set_title("log(1 + #transactions) by plan")
    ax[0].set_ylabel("")
    sns.boxplot(
        data=enriched.reset_index(),
        x="plan",
        y=np.log1p(enriched["total_completed_usd"]).values,
        ax=ax[1],
        order=["STANDARD", "SILVER", "GOLD"],
    )
    ax[1].set_title("log(1 + total completed USD) by plan")
    ax[1].set_ylabel("")
    _save_fig(fig, "activity_by_plan.png")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_eda(
    users: pd.DataFrame,
    devices: pd.DataFrame,
    notifs: pd.DataFrame,
    trx: pd.DataFrame,
) -> None:
    eda_users(users)
    eda_devices(devices, users)
    eda_notifications(notifs)
    eda_transactions(trx)
    eda_relationships(users, devices, notifs, trx)
    log.info("EDA done. Figures in %s, tables in %s", config.FIGURES_DIR, config.TABLES_DIR)
