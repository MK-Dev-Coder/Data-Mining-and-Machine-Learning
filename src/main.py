"""
End-to-end orchestrator: EDA → preprocessing → modelling.

Run from the project root with:

    python -m src.main
"""
from __future__ import annotations
import logging
import time

from . import config, data_loader, eda, preprocessing, engagement


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    configure_logging()
    log = logging.getLogger("main")

    t0 = time.time()
    log.info("Loading raw tables ...")
    tables = data_loader.load_all()
    users, devices, notifs, trx = (
        tables["users"], tables["devices"], tables["notifications"], tables["transactions"],
    )
    log.info(
        "Loaded users=%d, devices=%d, notifications=%d, transactions=%d (%.1fs)",
        len(users), len(devices), len(notifs), len(trx), time.time() - t0,
    )

    log.info("=== STEP 1: EDA ===")
    eda.run_eda(users, devices, notifs, trx)

    log.info("=== STEP 2: Preprocessing & feature engineering ===")
    X, y, meta = preprocessing.build_feature_table(users, devices, notifs, trx)
    X.to_parquet(config.TABLES_DIR / "feature_matrix.parquet") if False else \
        X.to_csv(config.TABLES_DIR / "feature_matrix.csv", index=False)
    y.to_csv(config.TABLES_DIR / "target.csv", index=False, header=["churned"])
    meta.to_csv(config.TABLES_DIR / "meta.csv", index=False)

    log.info("=== STEP 3: Engagement / churn modelling ===")
    engagement.run_modelling(X, y, meta)

    log.info("DONE in %.1fs. See outputs/ for figures, tables and the saved model.",
             time.time() - t0)


if __name__ == "__main__":
    main()
