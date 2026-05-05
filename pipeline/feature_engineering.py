"""
Phase 1 — Feature Engineering Pipeline
Run this from your project root: python pipeline/feature_engineering.py
"""

import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_PATH    = Path("data/raw/creditcard.csv")
STAGED_PATH = Path("data/staged/transactions_staged.csv")
REPORT_PATH = Path("data/staged/feature_report.txt")


def load_raw(path: Path) -> pd.DataFrame:
    log.info(f"Loading raw data from {path}")
    df = pd.read_csv(path)
    log.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped:
        log.warning(f"Dropped {dropped} rows with null values")
    else:
        log.info("No null values found")

    # Confirm expected columns exist
    required = {"Time", "Amount", "Class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    fraud_count = df["Class"].sum()
    log.info(f"Class balance — fraud: {fraud_count:,} ({fraud_count/len(df)*100:.2f}%)  "
             f"legit: {len(df)-fraud_count:,}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Engineering features...")

    # 1. Z-score normalize Amount (removes the raw dollar scale)
    mean_amt = df["Amount"].mean()
    std_amt  = df["Amount"].std()
    df["amount_zscore"] = (df["Amount"] - mean_amt) / std_amt
    log.info(f"  amount_zscore  — mean={mean_amt:.2f}, std={std_amt:.2f}")

    # 2. Normalize Time to hours elapsed (raw column is seconds from first transaction)
    df["time_hours"] = df["Time"] / 3600
    log.info(f"  time_hours     — range: {df['time_hours'].min():.1f}h "
             f"to {df['time_hours'].max():.1f}h")

    # 3. Rolling 24-transaction velocity (sum of amounts in last 24 rows)
    #    Proxy for "how much activity in recent window"
    df["velocity_24tx"] = (
        df["Amount"]
        .rolling(window=24, min_periods=1)
        .sum()
        .round(4)
    )
    log.info(f"  velocity_24tx  — mean={df['velocity_24tx'].mean():.2f}")

    # 4. Rolling 24-tx transaction count (how many txns in recent window)
    df["tx_count_24"] = (
        df["Amount"]
        .rolling(window=24, min_periods=1)
        .count()
        .astype(int)
    )

    # 5. Hour-of-day (cyclical — fraud spikes at night)
    df["hour_of_day"] = df["time_hours"].mod(24).astype(int)

    # 6. Amount bucket (log scale, avoids outlier domination)
    df["amount_log"] = np.log1p(df["Amount"]).round(4)

    # 7. Drop raw columns we've replaced
    df = df.drop(columns=["Time", "Amount"])

    log.info(f"Feature engineering complete — {df.shape[1]} columns total")
    return df


def save_staged(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    size_mb = path.stat().st_size / 1_000_000
    log.info(f"Saved staged data → {path}  ({size_mb:.1f} MB, {len(df):,} rows)")


def write_report(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "=" * 60,
        "FEATURE ENGINEERING REPORT",
        "=" * 60,
        f"Total rows       : {len(df):,}",
        f"Total features   : {df.shape[1]}",
        f"Fraud cases      : {df['Class'].sum():,}  ({df['Class'].mean()*100:.3f}%)",
        f"Legit cases      : {(df['Class']==0).sum():,}",
        "",
        "-- Feature summary --",
    ]
    for col in df.select_dtypes(include="number").columns:
        lines.append(
            f"  {col:<20} mean={df[col].mean():>10.4f}  "
            f"std={df[col].std():>10.4f}  "
            f"min={df[col].min():>10.4f}  "
            f"max={df[col].max():>10.4f}"
        )
    lines += ["", "Output saved to: data/staged/transactions_staged.csv", "=" * 60]

    path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Report written  → {path}")
    # Also print to terminal
    print("\n" + "\n".join(lines))


def run():
    log.info("---  Phase 1: Feature Engineering Pipeline  ---")

    df = load_raw(RAW_PATH)
    df = validate(df)
    df = engineer_features(df)
    save_staged(df, STAGED_PATH)
    write_report(df, REPORT_PATH)

    log.info("---  Done. Ready for Phase 2 (ML training).  ---")


if __name__ == "__main__":
    run()