"""
Step 2: cleaning, encoding and feature engineering.

Builds a single user-level feature matrix that captures everything we know
about a user as of the cutoff date. The cutoff is snapshot - LOOKBACK_DAYS
where snapshot is the last transaction date in the dataset. Transactions and
notifications occurring after the cutoff are discarded when building features
so that the target window does not leak into the inputs.

build_feature_table returns:
    X    : DataFrame with one row per eligible user (numeric + one-hot)
    y    : Series of {0,1} labels (1 == churned / unengaged)
    meta : DataFrame with bookkeeping columns (user_id, snapshot, cutoff,
           tenure_days, days_since_last_trx, ...) for downstream reporting.
"""
from __future__ import annotations
import logging
from typing import Tuple

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------
# 99.5th-percentile clip is enough to silence the obvious data-entry errors
# (max amount is 7.4e10 USD which is plainly impossible) without throwing
# away the long tail of legitimately-large transfers.
AMOUNT_CLIP_QUANTILE = 0.995


def _clean_users(users: pd.DataFrame) -> pd.DataFrame:
    df = users.copy()
    # Marketing-flag NaNs: per-product convention, we treat the absence of an
    # opt-in record as an implicit "not opted in" (0).
    for c in (
        "attributes_notifications_marketing_push",
        "attributes_notifications_marketing_email",
    ):
        df[c] = df[c].fillna(0).astype(int)
    df["age"] = 2019 - df["birth_year"]
    # Drop rows with implausible age. Keeps 100% of this dataset but is a
    # safety net if the same code is rerun on dirtier data.
    df = df[(df["age"] >= 14) & (df["age"] <= 100)].copy()
    return df


def _clean_transactions(trx: pd.DataFrame) -> pd.DataFrame:
    df = trx.copy()
    # negative or NaN amounts shouldn't happen; if they do, drop them
    df = df[df["amount_usd"].notna() & (df["amount_usd"] >= 0)].copy()
    # cap the absurd outliers
    cap = df["amount_usd"].quantile(AMOUNT_CLIP_QUANTILE)
    df["amount_usd_clipped"] = df["amount_usd"].clip(upper=cap)
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def _trx_features(trx_pre: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Aggregate transaction features per user using only data up to cutoff."""
    g = trx_pre.groupby("user_id")
    completed = trx_pre[trx_pre["transactions_state"] == "COMPLETED"]
    g_completed = completed.groupby("user_id")

    feat = pd.DataFrame(index=trx_pre["user_id"].unique())
    feat.index.name = "user_id"
    feat["trx_count"] = g.size()
    feat["trx_count_completed"] = g_completed.size()
    feat["trx_value_completed"] = g_completed["amount_usd_clipped"].sum()
    feat["trx_value_mean_completed"] = g_completed["amount_usd_clipped"].mean()
    feat["trx_value_max_completed"] = g_completed["amount_usd_clipped"].max()
    feat["trx_success_rate"] = feat["trx_count_completed"] / feat["trx_count"]

    feat["trx_n_distinct_currency"] = g["transactions_currency"].nunique()
    feat["trx_n_distinct_mcc"] = g["ea_merchant_mcc"].nunique()
    feat["trx_n_distinct_country"] = g["ea_merchant_country"].nunique()

    # share of each transaction type (behavioural profile)
    type_share = (
        trx_pre.groupby(["user_id", "transactions_type"]).size().unstack(fill_value=0)
    )
    type_share = type_share.div(type_share.sum(axis=1), axis=0).add_prefix("trx_share_")
    feat = feat.join(type_share, how="left")

    # share of inbound vs outbound
    direction = (
        trx_pre.dropna(subset=["direction"])
        .groupby(["user_id", "direction"])
        .size()
        .unstack(fill_value=0)
    )
    if not direction.empty:
        direction = direction.div(direction.sum(axis=1), axis=0).add_prefix("trx_dir_")
        feat = feat.join(direction, how="left")

    # Recency relative to the cutoff. This is "how many days inactive was the
    # user at the moment we ran the model"; the heuristic baseline thresholds
    # this directly. Using `cutoff` (not `snapshot`) is critical, otherwise
    # every user is trivially ≥ LOOKBACK_DAYS inactive and the threshold has
    # no discriminative power.
    last = g["created_date"].max()
    feat["days_since_last_trx"] = (cutoff - last).dt.days.clip(lower=0)
    feat["days_since_first_trx"] = (cutoff - g["created_date"].min()).dt.days.clip(lower=0)
    feat["trx_active_days"] = (
        g["created_date"].apply(lambda s: s.dt.normalize().nunique())
    )
    return feat.reset_index()


def _notif_features(notifs_pre: pd.DataFrame) -> pd.DataFrame:
    g = notifs_pre.groupby("user_id")
    feat = pd.DataFrame(index=notifs_pre["user_id"].unique())
    feat.index.name = "user_id"
    feat["notif_count"] = g.size()
    feat["notif_n_sent"] = g["status"].apply(lambda s: (s == "SENT").sum())
    feat["notif_n_failed"] = g["status"].apply(lambda s: (s == "FAILED").sum())
    feat["notif_send_rate"] = feat["notif_n_sent"] / feat["notif_count"].replace(0, np.nan)

    by_channel = (
        notifs_pre.groupby(["user_id", "channel"]).size().unstack(fill_value=0)
    )
    by_channel = by_channel.add_prefix("notif_ch_")
    feat = feat.join(by_channel, how="left")

    # any "REENGAGEMENT_*" notification means the business already flagged
    # this user as at risk; we keep it as an explicit feature
    re_eng = notifs_pre[notifs_pre["reason"].str.contains("REENGAGEMENT", na=False)]
    feat["notif_n_reengagement"] = re_eng.groupby("user_id").size()
    feat["notif_n_reengagement"] = feat["notif_n_reengagement"].fillna(0)

    return feat.reset_index()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_feature_table(
    users: pd.DataFrame,
    devices: pd.DataFrame,
    notifs: pd.DataFrame,
    trx: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    users = _clean_users(users)
    trx = _clean_transactions(trx)

    snapshot = trx["created_date"].max().normalize()
    cutoff = snapshot - pd.Timedelta(days=config.LOOKBACK_DAYS)
    log.info("Snapshot=%s, Cutoff=%s, Lookback=%dd",
             snapshot.date(), cutoff.date(), config.LOOKBACK_DAYS)

    # --- target ------------------------------------------------------------
    in_window = trx[
        (trx["created_date"] > cutoff)
        & (trx["created_date"] <= snapshot)
        & (trx["transactions_state"] == "COMPLETED")
    ]
    engaged_users = set(in_window["user_id"].unique())

    # --- pre-cutoff slices used for features (no leakage) -----------------
    trx_pre = trx[trx["created_date"] <= cutoff]
    notifs_pre = notifs[notifs["created_date"] <= cutoff]

    # --- feature matrices --------------------------------------------------
    trx_feat = _trx_features(trx_pre, cutoff=cutoff)
    notif_feat = _notif_features(notifs_pre)

    # --- merge with user-level features -----------------------------------
    df = (
        users.merge(devices, on="user_id", how="left")
        .merge(trx_feat, on="user_id", how="left")
        .merge(notif_feat, on="user_id", how="left")
    )

    # users with zero pre-cutoff activity get zeros, not NaNs, on count fields
    count_cols = [c for c in df.columns if c.startswith(("trx_", "notif_"))]
    for c in count_cols:
        if df[c].dtype.kind in "fi":
            df[c] = df[c].fillna(0)

    # tenure before cutoff
    df["tenure_days"] = (cutoff - df["created_date"]).dt.days

    # last-active recency (cap at 999 if we never saw a transaction)
    df["days_since_last_trx"] = df["days_since_last_trx"].fillna(999)
    df["days_since_first_trx"] = df["days_since_first_trx"].fillna(999)
    df["trx_success_rate"] = df["trx_success_rate"].fillna(0)

    # --- eligibility -------------------------------------------------------
    eligible = df["tenure_days"] >= config.MIN_TENURE_DAYS
    log.info("Eligible users: %d / %d (>= %d days tenure)",
             int(eligible.sum()), len(df), config.MIN_TENURE_DAYS)
    df = df[eligible].copy()

    # --- target column -----------------------------------------------------
    df["churned"] = (~df["user_id"].isin(engaged_users)).astype(int)
    log.info("Class balance: churn=%d (%.1f%%), engaged=%d (%.1f%%)",
             int(df["churned"].sum()), 100 * df["churned"].mean(),
             int((1 - df["churned"]).sum()), 100 * (1 - df["churned"].mean()))

    # --- encoding ----------------------------------------------------------
    # plan + brand: low cardinality → one-hot
    df = pd.get_dummies(df, columns=["plan", "brand"], drop_first=False)

    # country: keep only top-N, group rest as "OTHER"
    top_countries = df["country"].value_counts().nlargest(15).index
    df["country_grp"] = df["country"].where(df["country"].isin(top_countries), "OTHER")
    df = pd.get_dummies(df, columns=["country_grp"], prefix="country")

    # drop columns that are not features (identifiers / raw dates / leakage)
    drop_cols = [
        "user_id", "city", "country", "created_date", "birth_year",
    ]
    meta_cols = ["user_id", "country", "plan_GOLD", "plan_SILVER", "plan_STANDARD",
                 "tenure_days", "days_since_last_trx", "trx_count", "churned"]

    meta = df.reindex(columns=[c for c in meta_cols if c in df.columns]).copy()
    meta["snapshot"] = snapshot
    meta["cutoff"] = cutoff
    meta["user_id"] = df["user_id"].values

    y = df["churned"].astype(int)
    X = df.drop(columns=drop_cols + ["churned"])

    # cast booleans (from get_dummies) to int for downstream sklearn
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)

    log.info("Feature matrix: %d rows × %d cols", *X.shape)
    return X, y, meta
