"""
Loads the four raw CSV tables and returns them as DataFrames.

The transactions table is split across three files; we concatenate them and
parse the date columns once at the source so downstream code never has to
worry about strings.
"""
from __future__ import annotations
import pandas as pd

from . import config


def load_users() -> pd.DataFrame:
    return pd.read_csv(config.USERS_CSV, parse_dates=["created_date"])


def load_devices() -> pd.DataFrame:
    return pd.read_csv(config.DEVICES_CSV)


def load_notifications() -> pd.DataFrame:
    return pd.read_csv(config.NOTIFICATIONS_CSV, parse_dates=["created_date"])


def load_transactions() -> pd.DataFrame:
    parts = [
        pd.read_csv(p, parse_dates=["created_date"]) for p in config.TRANSACTION_CSVS
    ]
    return pd.concat(parts, ignore_index=True)


def load_all() -> dict[str, pd.DataFrame]:
    """Load all four tables and return them keyed by name."""
    return {
        "users": load_users(),
        "devices": load_devices(),
        "notifications": load_notifications(),
        "transactions": load_transactions(),
    }
