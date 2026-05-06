"""
Phase 2 — Fraud Classifier Training
Run from project root: python ml/train_fraud_classifier.py

Outputs:
  ml/models/fraud_model.pkl        — pickled XGBoost model
  ml/models/fraud_model.onnx       — ONNX export for Azure ML later
  ml/models/feature_names.txt      — ordered feature list FastAPI will use
  ml/reports/evaluation_report.txt — AUC, precision, recall, F1, confusion matrix
"""

import pandas as pd
import numpy as np
import pickle
import logging
import os
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
STAGED_PATH  = Path("data/staged/transactions_staged.csv")
MODEL_DIR    = Path("ml/models")
REPORT_DIR   = Path("ml/reports")
MODEL_PATH   = MODEL_DIR / "fraud_model.pkl"
ONNX_PATH    = MODEL_DIR / "fraud_model.onnx"
FEATURES_PATH= MODEL_DIR / "feature_names.txt"
REPORT_PATH  = REPORT_DIR / "evaluation_report.txt"

# ── Feature columns (V1-V28 + engineered features, NOT Class/time_hours/tx_count_24)
FEATURE_COLS = (
    [f"V{i}" for i in range(1, 29)]
    + ["amount_zscore", "velocity_24tx", "hour_of_day", "amount_log"]
)
TARGET_COL = "Class"


def load_data() -> pd.DataFrame:
    log.info(f"Loading staged data from {STAGED_PATH}")
    df = pd.read_csv(STAGED_PATH)
    log.info(f"Loaded {len(df):,} rows")

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in staged data: {missing}")

    return df


def split_data(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    fraud_count = int(y.sum())
    legit_count = int((y == 0).sum())
    ratio = legit_count // fraud_count
    log.info(f"Class balance — fraud: {fraud_count:,}  legit: {legit_count:,}  ratio: 1:{ratio}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y        # preserve fraud ratio in both splits
    )
    log.info(f"Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    log.info(f"Fraud in train: {y_train.sum():,}  |  Fraud in test: {y_test.sum():,}")

    return X_train, X_test, y_train, y_test, ratio


def train(X_train, y_train, scale_pos_weight: int):
    log.info("Training XGBoost classifier...")
    log.info(f"scale_pos_weight={scale_pos_weight} (handles class imbalance)")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,  # critical for imbalanced fraud data
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,          # use all CPU cores
        verbosity=0,
    )

    # 5-fold cross-validation on training set to check for overfitting
    log.info("Running 5-fold cross-validation on training set...")
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="roc_auc",
        n_jobs=-1
    )
    log.info(f"CV AUC scores: {[round(s, 4) for s in cv_scores]}")
    log.info(f"CV AUC mean={cv_scores.mean():.4f}  std={cv_scores.std():.4f}")

    # Final fit on full training set
    model.fit(X_train, y_train)
    log.info("Training complete")
    return model, cv_scores


def evaluate(model, X_test, y_test):
    log.info("Evaluating on held-out test set...")

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc       = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    cm        = confusion_matrix(y_test, y_pred)

    log.info(f"AUC-ROC   : {auc:.4f}")
    log.info(f"Precision : {precision:.4f}")
    log.info(f"Recall    : {recall:.4f}")
    log.info(f"F1 Score  : {f1:.4f}")

    tn, fp, fn, tp = cm.ravel()
    log.info(f"Confusion matrix — TP:{tp}  FP:{fp}  TN:{tn}  FN:{fn}")

    return {
        "auc": auc, "precision": precision,
        "recall": recall, "f1": f1,
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
        "y_prob": y_prob, "y_pred": y_pred
    }


def get_feature_importance(model) -> pd.DataFrame:
    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    return importance


def save_model(model) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Pickle (used by FastAPI /predict)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    log.info(f"Saved pickle  -> {MODEL_PATH}")

    # 2. Feature names (FastAPI uses this to validate incoming payloads)
    FEATURES_PATH.write_text("\n".join(FEATURE_COLS), encoding="utf-8")
    log.info(f"Saved features -> {FEATURES_PATH}")

    # 3. ONNX export (for Azure ML endpoint later)
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        initial_type = [("float_input", FloatTensorType([None, len(FEATURE_COLS)]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        with open(ONNX_PATH, "wb") as f:
            f.write(onnx_model.SerializeToString())
        log.info(f"Saved ONNX    -> {ONNX_PATH}")
    except ImportError:
        log.warning("skl2onnx not installed — skipping ONNX export.")
        log.warning("Install later with: pip install skl2onnx")


def write_report(metrics: dict, cv_scores, importance: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 60,
        "FRAUD CLASSIFIER EVALUATION REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: XGBoost  |  Features: {len(FEATURE_COLS)}",
        "=" * 60,
        "",
        "-- Test Set Metrics --",
        f"  AUC-ROC   : {metrics['auc']:.4f}",
        f"  Precision : {metrics['precision']:.4f}",
        f"  Recall    : {metrics['recall']:.4f}",
        f"  F1 Score  : {metrics['f1']:.4f}",
        "",
        "-- Confusion Matrix --",
        f"  True Positives  (caught fraud)    : {metrics['tp']:>6}",
        f"  False Positives (false alarms)    : {metrics['fp']:>6}",
        f"  True Negatives  (correct legit)   : {metrics['tn']:>6}",
        f"  False Negatives (missed fraud)    : {metrics['fn']:>6}",
        "",
        "-- Cross-Validation (5-fold, training set) --",
        f"  Scores : {[round(s, 4) for s in cv_scores]}",
        f"  Mean   : {cv_scores.mean():.4f}",
        f"  Std    : {cv_scores.std():.4f}",
        "",
        "-- Top 10 Feature Importances --",
    ]

    for _, row in importance.head(10).iterrows():
        bar = "#" * int(row["importance"] * 50)
        lines.append(f"  {row['feature']:<20} {row['importance']:.4f}  {bar}")

    lines += [
        "",
        "-- Output Files --",
        f"  Model (pickle) : {MODEL_PATH}",
        f"  Model (ONNX)   : {ONNX_PATH}",
        f"  Feature names  : {FEATURES_PATH}",
        "=" * 60,
    ]

    report_text = "\n".join(lines)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print("\n" + report_text)
    log.info(f"Report saved -> {REPORT_PATH}")


def upload_to_blob() -> None:
    """
    Upload model to Azure Blob Storage.
    Skips gracefully if Azure credentials aren't configured yet.
    """
    try:
        from azure.storage.blob import BlobServiceClient
        from dotenv import load_dotenv
        load_dotenv()

        conn_str = os.getenv("STORAGE_CONNECTION_STRING")
        if not conn_str:
            log.warning("STORAGE_CONNECTION_STRING not set — skipping blob upload.")
            log.warning("Add it to your .env file when Azure is fully configured.")
            return

        client = BlobServiceClient.from_connection_string(conn_str)
        container = client.get_container_client("models")

        for path in [MODEL_PATH, FEATURES_PATH]:
            blob_name = path.name
            with open(path, "rb") as f:
                container.upload_blob(blob_name, f, overwrite=True)
            log.info(f"Uploaded {blob_name} -> Azure Blob (models container)")

    except ImportError:
        log.warning("azure-storage-blob not installed — skipping blob upload.")
    except Exception as e:
        log.warning(f"Blob upload failed: {e}")
        log.warning("Model is still saved locally — Azure upload can be retried later.")


def run():
    log.info("---  Phase 2: Fraud Classifier Training  ---")

    df                                    = load_data()
    X_train, X_test, y_train, y_test, ratio = split_data(df)
    model, cv_scores                      = train(X_train, y_train, scale_pos_weight=ratio)
    metrics                               = evaluate(model, X_test, y_test)
    importance                            = get_feature_importance(model)

    save_model(model)
    write_report(metrics, cv_scores, importance)
    upload_to_blob()

    log.info("---  Done. Ready for Phase 3 (FastAPI /predict endpoint).  ---")


if __name__ == "__main__":
    run()