"""
Phase 3 — FastAPI Backend
Run from project root: uvicorn backend.main:app --reload

Endpoints:
  POST /ingest        — validate + store transaction/portfolio data
  POST /predict       — fraud score from XGBoost model
  GET  /search        — vector search over regulatory docs (stub -> RAG in Phase 4)
  POST /agent/query   — route natural language question to correct agent (stub -> Phase 4)

Docs available at: http://127.0.0.1:8000/docs
"""

import os
import pickle
import logging
import sqlite3
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("finrisk")

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH    = Path(r"C:\Users\Rayapu reddy\Desktop\Finance-Fraud\ml\models\fraud_model.pkl")
FEATURES_PATH = Path(r"C:\Users\Rayapu reddy\Desktop\Finance-Fraud\ml\models\feature_names.txt")
DB_PATH       = Path(r"C:\Users\Rayapu reddy\Desktop\Finance-Fraud\data\finrisk.db")
FORECAST_PATH = Path(r"C:\Users\Rayapu reddy\Desktop\Finance-Fraud\data\staged\portfolio_forecast.csv")

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Global model state ─────────────────────────────────────────────────────────
_model        = None
_feature_cols = None


def load_model():
    global _model, _feature_cols
    try:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        _feature_cols = FEATURES_PATH.read_text(encoding="utf-8").strip().splitlines()
        log.info(f"Loaded fraud model — {len(_feature_cols)} features")
    except FileNotFoundError as e:
        log.error(f"Model file not found: {e}")
        log.error("Run ml/train_fraud_classifier.py first")
        raise


def init_db():
    """Create SQLite tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            tx_id       TEXT UNIQUE,
            amount      REAL,
            merchant    TEXT,
            timestamp   TEXT,
            features    TEXT,   -- JSON blob of V1-V28 + engineered features
            fraud_score REAL,
            label       INTEGER,
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date   TEXT,
            total_value_inr REAL,
            asset_class     TEXT,
            weight          REAL,
            created_at      TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS agent_queries (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            question    TEXT,
            agent_used  TEXT,
            answer      TEXT,
            latency_ms  INTEGER,
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    log.info(f"SQLite DB ready -> {DB_PATH}")


# ── Lifespan (replaces @app.on_event deprecated in FastAPI 0.95+) ──────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting FinRisk API...")
    load_model()
    init_db()
    log.info("API ready")
    yield
    log.info("Shutting down FinRisk API")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FinRisk AI Platform",
    description="Financial Risk & Advisory API — Fraud Detection, Portfolio Analytics, Regulatory Search, Multi-Agent Query",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten this when deploying
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class TransactionIngest(BaseModel):
    tx_id    : str              = Field(..., example="TX-94821")
    amount   : float            = Field(..., gt=0, example=2400.50)
    merchant : str              = Field(..., example="Online Electronics")
    timestamp: str              = Field(..., example="2025-05-01T14:30:00")
    # V1-V28 PCA features from the card processor (optional for ingest-only calls)
    features : Optional[dict]   = Field(None, description="V1-V28 + engineered features")

    @validator("timestamp")
    def validate_timestamp(cls, v):
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError("timestamp must be ISO 8601 format: 2025-05-01T14:30:00")
        return v


class PortfolioIngest(BaseModel):
    snapshot_date   : str   = Field(..., example="2025-05-01")
    total_value_inr : float = Field(..., gt=0, example=4820000000.0)
    holdings        : list[dict] = Field(..., example=[
        {"asset": "HDFCBANK.NS", "weight": 0.182},
        {"asset": "INFY.NS",     "weight": 0.147},
    ])


class IngestRequest(BaseModel):
    data_type  : str                        = Field(..., example="transaction")  # "transaction" or "portfolio"
    transaction: Optional[TransactionIngest] = None
    portfolio  : Optional[PortfolioIngest]   = None

    @validator("data_type")
    def validate_data_type(cls, v):
        if v not in ("transaction", "portfolio"):
            raise ValueError("data_type must be 'transaction' or 'portfolio'")
        return v


class PredictRequest(BaseModel):
    """
    Send all 32 features. Feature names match feature_names.txt exactly.
    V1-V28 come from your card processor (PCA-transformed).
    The engineered features come from your pipeline output.
    """
    # PCA features
    V1 : float; V2 : float; V3 : float; V4 : float; V5 : float
    V6 : float; V7 : float; V8 : float; V9 : float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    # Engineered features from pipeline
    amount_zscore : float = Field(..., example=1.5)
    velocity_24tx : float = Field(..., example=3200.0)
    hour_of_day   : int   = Field(..., ge=0, le=23, example=2)
    amount_log    : float = Field(..., example=7.8)
    # Optional metadata stored alongside the score
    tx_id         : Optional[str] = None


class AgentQueryRequest(BaseModel):
    question   : str  = Field(..., example="What was the average fraud score last week?")
    session_id : Optional[str] = None   # for multi-turn context in Phase 4


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 1 — POST /ingest
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/ingest", tags=["Data"])
def ingest(req: IngestRequest):
    """
    Accept and store transaction or portfolio data.
    Validates with Pydantic, writes to SQLite (swap to Azure SQL in Phase 5).
    """
    log.info(f"POST /ingest — data_type={req.data_type}")

    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()

        if req.data_type == "transaction":
            if not req.transaction:
                raise HTTPException(status_code=422, detail="transaction payload required")
            tx = req.transaction
            import json
            cur.execute("""
                INSERT OR REPLACE INTO transactions
                    (tx_id, amount, merchant, timestamp, features)
                VALUES (?, ?, ?, ?, ?)
            """, (
                tx.tx_id,
                tx.amount,
                tx.merchant,
                tx.timestamp,
                json.dumps(tx.features) if tx.features else None,
            ))
            conn.commit()
            log.info(f"Ingested transaction {tx.tx_id} — amount={tx.amount}")
            return {
                "status"   : "ok",
                "ingested" : "transaction",
                "tx_id"    : tx.tx_id,
                "stored_at": datetime.utcnow().isoformat(),
            }

        elif req.data_type == "portfolio":
            if not req.portfolio:
                raise HTTPException(status_code=422, detail="portfolio payload required")
            pf = req.portfolio
            for holding in pf.holdings:
                cur.execute("""
                    INSERT INTO portfolio_snapshots
                        (snapshot_date, total_value_inr, asset_class, weight)
                    VALUES (?, ?, ?, ?)
                """, (
                    pf.snapshot_date,
                    pf.total_value_inr,
                    holding.get("asset", "unknown"),
                    holding.get("weight", 0.0),
                ))
            conn.commit()
            log.info(f"Ingested portfolio snapshot {pf.snapshot_date} — {len(pf.holdings)} holdings")
            return {
                "status"      : "ok",
                "ingested"    : "portfolio",
                "snapshot_date": pf.snapshot_date,
                "holdings_count": len(pf.holdings),
                "stored_at"   : datetime.utcnow().isoformat(),
            }

    except sqlite3.IntegrityError as e:
        log.warning(f"Duplicate entry: {e}")
        raise HTTPException(status_code=409, detail=f"Duplicate tx_id: {req.transaction.tx_id}")
    except Exception as e:
        log.error(f"Ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 2 — POST /predict
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/predict", tags=["ML"])
def predict(req: PredictRequest):
    """
    Run fraud classifier on a transaction's feature vector.
    Returns fraud_score (0-1), label (0=legit, 1=fraud), and risk_level.
    """
    log.info(f"POST /predict — tx_id={req.tx_id or 'anonymous'}")

    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Build feature vector in the exact order the model was trained on
        feature_values = [getattr(req, col) for col in _feature_cols]
        X = np.array(feature_values).reshape(1, -1)

        fraud_score = float(_model.predict_proba(X)[0][1])
        label       = int(fraud_score >= 0.5)

        # Risk tier for the dashboard
        if fraud_score >= 0.80:
            risk_level = "HIGH"
        elif fraud_score >= 0.50:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Persist score back to DB if we have a tx_id
        if req.tx_id:
            try:
                conn = sqlite3.connect(DB_PATH)
                conn.execute(
                    "UPDATE transactions SET fraud_score=?, label=? WHERE tx_id=?",
                    (fraud_score, label, req.tx_id)
                )
                conn.commit()
                conn.close()
            except Exception as db_err:
                log.warning(f"Could not update fraud score in DB: {db_err}")

        log.info(f"Prediction — score={fraud_score:.4f} label={label} risk={risk_level}")

        return {
            "tx_id"      : req.tx_id,
            "fraud_score": round(fraud_score, 4),
            "label"      : label,
            "risk_level" : risk_level,
            "model"      : "XGBoost v1.0",
            "predicted_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        log.error(f"Predict error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 3 — GET /search
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/search", tags=["Regulatory"])
def search(
    q      : str           = Query(..., min_length=3, example="SEBI AIF Category II disclosure"),
    source : Optional[str] = Query(None, example="SEBI"),   # filter by source
    top_k  : int           = Query(5, ge=1, le=20),
):
    """
    Search regulatory documents.
    Phase 3: returns mock ranked results.
    Phase 4: replaced with real FAISS / Azure Cognitive Search RAG retrieval.
    """
    log.info(f"GET /search — q='{q}' source={source} top_k={top_k}")

    # ── Stub results (replaced by real vector search in Phase 4) ──────────────
    mock_docs = [
        {
            "rank"           : 1,
            "title"          : "SEBI (AIF) Regulations 2012 — Circular SEBI/LAD-NRO/GN/2021-22",
            "source"         : "SEBI",
            "relevance_score": 0.96,
            "chunk"          : "AIF Category II funds must file quarterly reports within 10 days of quarter-end including portfolio details, NAV declarations, and investor-wise exposure limits.",
            "page"           : 14,
        },
        {
            "rank"           : 2,
            "title"          : "Master Circular for Alternative Investment Funds — Reporting Requirements",
            "source"         : "SEBI",
            "relevance_score": 0.91,
            "chunk"          : "Annual audited accounts must be submitted within 180 days from the end of each financial year.",
            "page"           : 8,
        },
        {
            "rank"           : 3,
            "title"          : "RBI Guidelines on Foreign Investment in AIF Structures",
            "source"         : "RBI",
            "relevance_score": 0.78,
            "chunk"          : "Foreign investors in Category II AIFs are subject to FEMA regulations and must report exposure limits to RBI on a quarterly basis.",
            "page"           : 22,
        },
    ]

    # Filter by source if provided
    results = [d for d in mock_docs if not source or d["source"] == source][:top_k]

    return {
        "query"       : q,
        "source_filter": source,
        "total_results": len(results),
        "results"     : results,
        "backend"     : "mock — replace with FAISS/Azure Cognitive Search in Phase 4",
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 4 — POST /agent/query
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/agent/query", tags=["Agents"])
def agent_query(req: AgentQueryRequest):
    """
    Route a natural language question to the correct agent.
    Phase 3: intent detection + stub responses with DB lookups where possible.
    Phase 4: replaced with real LangChain/CrewAI agents.
    """
    import time
    start = time.time()

    log.info(f"POST /agent/query — question='{req.question[:60]}...'")

    question_lower = req.question.lower()

    # ── Intent routing ─────────────────────────────────────────────────────────
    sql_keywords  = ["average", "count", "total", "how many", "last week",
                     "last month", "sum", "fraud score", "transaction"]
    rag_keywords  = ["sebi", "rbi", "regulation", "compliance", "basel",
                     "fema", "guideline", "circular", "disclosure", "aif"]
    ml_keywords   = ["explain", "why", "score", "probability", "feature",
                     "model", "predict", "anomaly"]

    if any(k in question_lower for k in rag_keywords):
        agent_used = "rag_agent"
        answer     = _stub_rag_answer(req.question)
    elif any(k in question_lower for k in sql_keywords):
        agent_used = "sql_agent"
        answer     = _stub_sql_answer(req.question)
    elif any(k in question_lower for k in ml_keywords):
        agent_used = "ml_expert_agent"
        answer     = _stub_ml_answer(req.question)
    else:
        agent_used = "orchestrator"
        answer     = (
            f"I received your question: '{req.question}'. "
            "I'm routing this to the most appropriate agent. "
            "Full agent capabilities will be available in Phase 4."
        )

    latency_ms = int((time.time() - start) * 1000)

    # Store query in DB for the Agent Insights Power BI tab
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO agent_queries (question, agent_used, answer, latency_ms) VALUES (?,?,?,?)",
            (req.question, agent_used, answer, latency_ms)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning(f"Could not store agent query: {e}")

    log.info(f"Agent response — agent={agent_used} latency={latency_ms}ms")

    return {
        "question"   : req.question,
        "agent_used" : agent_used,
        "answer"     : answer,
        "latency_ms" : latency_ms,
        "session_id" : req.session_id,
        "note"       : "Stub response — real LangChain agents wired in Phase 4",
    }


# ── Agent stubs (replaced by real agents in Phase 4) ──────────────────────────

def _stub_sql_answer(question: str) -> str:
    """Query SQLite for real numbers where possible."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()

        cur.execute("SELECT COUNT(*), AVG(fraud_score) FROM transactions WHERE fraud_score IS NOT NULL")
        row = cur.fetchone()
        conn.close()

        count = row[0] or 0
        avg   = round(row[1], 4) if row[1] else "N/A"

        return (
            f"Based on the transactions database: {count} transactions have been scored. "
            f"Average fraud score: {avg}. "
            f"[Phase 4 upgrade: SQL Agent will generate dynamic queries for any question.]"
        )
    except Exception as e:
        return f"SQL Agent stub — DB query failed: {e}. Phase 4 will wire LangChain SQL agent."


def _stub_rag_answer(question: str) -> str:
    return (
        "Based on SEBI AIF Regulations (Circular SEBI/LAD-NRO/GN/2021-22): "
        "Category II AIF funds must file quarterly reports within 10 days of quarter-end. "
        "Key disclosures include portfolio details, NAV declarations, and investor-wise exposure limits. "
        "Annual audited accounts must be submitted within 180 days from fiscal year end. "
        "[Phase 4 upgrade: RAG Agent will retrieve from indexed regulatory PDFs via FAISS/Azure Cognitive Search.]"
    )


def _stub_ml_answer(question: str) -> str:
    return (
        "This transaction scored 0.87 (HIGH risk). "
        "The top contributing features were: amount_zscore=4.1 (amount is 4x above normal), "
        "velocity_24tx=8200 (12x average transaction velocity in recent window), "
        "and hour_of_day=3 (transaction occurred at 3am, a high-risk window). "
        "[Phase 4 upgrade: ML Expert Agent will explain any live prediction using SHAP values.]"
    )


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
def health():
    return {
        "status"      : "ok",
        "model_loaded": _model is not None,
        "features"    : len(_feature_cols) if _feature_cols else 0,
        "db"          : str(DB_PATH),
        "timestamp"   : datetime.utcnow().isoformat(),
    }


@app.get("/forecast/portfolio", tags=["ML"])
def get_portfolio_forecast(days: int = Query(30, ge=7, le=90)):
    """Return the Prophet portfolio forecast from staged CSV."""
    try:
        df = pd.read_csv(FORECAST_PATH)
        df["ds"] = pd.to_datetime(df["ds"])

        # Return last 60 days historical + forecast days
        today     = df["ds"].max()
        cutoff    = today - pd.Timedelta(days=60)
        result_df = df[df["ds"] >= cutoff].copy()
        result_df["ds"] = result_df["ds"].dt.strftime("%Y-%m-%d")

        return {
            "forecast_days": days,
            "rows"         : len(result_df),
            "data"         : result_df.to_dict(orient="records"),
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Portfolio forecast not found. Run ml/train_forecaster.py first."
        )