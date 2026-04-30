"""
Project-wide paths and constants.

Edit DATA_DIR if the raw CSVs live somewhere else; everything else is derived.
"""
from __future__ import annotations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- input data --------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "churn_train"
USERS_CSV = DATA_DIR / "users.csv"
DEVICES_CSV = DATA_DIR / "devices.csv"
NOTIFICATIONS_CSV = DATA_DIR / "notifications.csv"
TRANSACTION_CSVS = [DATA_DIR / f"transactions_{i}.csv" for i in (1, 2, 3)]

# --- output artefacts --------------------------------------------------------
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"

for _p in (FIGURES_DIR, TABLES_DIR, MODELS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# --- engagement / churn definition ------------------------------------------
# Number of days to look back from the snapshot date to define the
# "engagement window". A user is engaged if they have >=1 completed
# transaction in (snapshot - LOOKBACK_DAYS, snapshot].
LOOKBACK_DAYS = 28

# Minimum tenure (days between user registration and the cutoff date) below
# which a user is dropped from the modelling sample. Without this, brand-new
# users are unfairly labelled as churn before they had a chance to be active.
MIN_TENURE_DAYS = 28

# Reproducibility
RANDOM_STATE = 42
