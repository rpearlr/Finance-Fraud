"""
FinRisk AI Platform — FastAPI Backend (fully connected to RAG pipeline)
Run from project root: uvicorn backend.main:app --reload

Endpoints:
  POST /ingest          — validate + store transaction/portfolio data
  POST /predict         — fraud score from XGBoost model
  GET  /search          — real FAISS vector search over regulatory docs
  POST /agent/query     — routes question to real RAG / SQL / ML agent
  GET  /health          — system health check
  GET  /forecast/portfolio — Prophet forecast data
"""

import os
import json
import time
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
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
from agents.fraud_agent import FraudExpertAgent
from agents.rag_pipeline import DocumentAssistantAgent
load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("finrisk")

# ── Paths (from .env or sensible defaults relative to project root) ────────────
MODEL_PATH    = Path(r"C:\Users\User\Desktop\fin-prj\ml\models\fraud_model.pkl")
FEATURES_PATH = Path( r"C:\Users\User\Desktop\fin-prj\ml\models\feature_names.txt")
DB_PATH       = Path(r"C:\Users\User\Desktop\fin-prj\data\finrisk.db")
FORECAST_PATH = Path(r"C:\Users\User\Desktop\fin-prj\data\staged\portfolio_forecast.csv")

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Global state ───────────────────────────────────────────────────────────────
_model        = None   # XGBoost fraud classifier
_feature_cols = None   # ordered list of feature names

_fraud_agent = None
_rag_agent   = None
# ══════════════════════════════════════════════════════════════════════════════
# STARTUP HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def load_model():
    global _model, _feature_cols, _fraud_agent, _rag_agent

    # existing model loading — keep this unchanged
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
    _feature_cols = FEATURES_PATH.read_text(encoding="utf-8").strip().splitlines()
    log.info(f"Loaded fraud model — {len(_feature_cols)} features")

    # initialise agents
    _fraud_agent = FraudExpertAgent()
    _rag_agent   = DocumentAssistantAgent()
    log.info("Agents initialised")



def init_db() -> None:
    """Create all SQLite tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS transactions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            tx_id       TEXT UNIQUE,
            amount      REAL,
            merchant    TEXT,
            timestamp   TEXT,
            features    TEXT,
            fraud_score REAL,
            label       INTEGER,
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date   TEXT,
            total_value_inr REAL,
            asset_class     TEXT,
            weight          REAL,
            created_at      TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS agent_queries (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            question    TEXT,
            agent_used  TEXT,
            answer      TEXT,
            sources     TEXT,
            chunks_used INTEGER,
            tokens_used INTEGER,
            latency_ms  INTEGER,
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    conn.close()
    log.info(f"SQLite DB ready — {DB_PATH}")


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting FinRisk API...")
    load_model()
    init_db()
    log.info("API ready ✓")
    yield
    log.info("Shutting down FinRisk API")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FinRisk AI Platform",
    description=(
        "Financial Risk & Advisory API — "
        "Fraud Detection · Portfolio Analytics · Regulatory RAG Search · Multi-Agent Query"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class TransactionIngest(BaseModel):
    tx_id    : str            = Field(..., examples=["TX-94821"])
    amount   : float          = Field(..., gt=0, examples=[2400.50])
    merchant : str            = Field(..., examples=["Online Electronics"])
    timestamp: str            = Field(..., examples=["2025-05-01T14:30:00"])
    features : Optional[dict] = Field(None, description="V1-V28 + engineered features")

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError("timestamp must be ISO 8601: 2025-05-01T14:30:00")
        return v


class PortfolioIngest(BaseModel):
    snapshot_date   : str        = Field(..., examples=["2025-05-01"])
    total_value_inr : float      = Field(..., gt=0, examples=[4820000000.0])
    holdings        : list[dict] = Field(..., examples=[[
        {"asset": "HDFCBANK.NS", "weight": 0.182},
        {"asset": "INFY.NS",     "weight": 0.147},
    ]])


class IngestRequest(BaseModel):
    data_type  : str                         = Field(..., examples=["transaction"])
    transaction: Optional[TransactionIngest] = None
    portfolio  : Optional[PortfolioIngest]   = None

    @field_validator("data_type")
    @classmethod
    def validate_data_type(cls, v: str) -> str:
        if v not in ("transaction", "portfolio"):
            raise ValueError("data_type must be 'transaction' or 'portfolio'")
        return v


class PredictRequest(BaseModel):
    # PCA features V1–V28
    V1 : float; V2 : float; V3 : float; V4 : float; V5 : float
    V6 : float; V7 : float; V8 : float; V9 : float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    # Engineered features
    amount_zscore : float = Field(..., examples=[1.5])
    velocity_24tx : float = Field(..., examples=[3200.0])
    hour_of_day   : int   = Field(..., ge=0, le=23, examples=[2])
    amount_log    : float = Field(..., examples=[7.8])
    # Optional metadata
    tx_id: Optional[str] = None


class AgentQueryRequest(BaseModel):
    question  : str           = Field(..., examples=["What are SEBI AIF disclosure requirements?"])
    session_id: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _detect_source_filter(question_lower: str) -> Optional[str]:
    """Infer regulatory source from question keywords."""
    if any(k in question_lower for k in ["sebi", "aif", "mutual fund", "securities"]):
        return "SEBI"
    if any(k in question_lower for k in ["rbi", "reserve bank", "monetary", "nbfc"]):
        return "RBI"
    if any(k in question_lower for k in ["basel", "capital adequacy", "cet1", "tier 1"]):
        return "Basel"
    if any(k in question_lower for k in ["fema", "foreign exchange", "forex", "fdi"]):
        return "FEMA"
    if any(k in question_lower for k in ["irdai", "insurance", "insurer"]):
        return "IRDAI"
    return None


def _rag_answer(question: str) -> dict:
    """
    Run the full RAG pipeline: retrieve chunks → generate grounded answer.
    Returns the full result dict from DocumentAssistantAgent.run().
    """
    if _rag_agent is None:
        return {
            "answer"     : "RAG agent could not be loaded. Check agents/rag_pipeline.py.",
            "sources"    : [],
            "chunks_used": 0,
            "tokens_used": 0,
            "latency_ms" : 0,
        }

    if not _rag_agent._ready:
        return {
            "answer"     : (
                "The regulatory document index has not been built yet. "
                "Run this command from the project root and restart the API:\n"
                "  python agents/rag_pipeline.py index"
            ),
            "sources"    : [],
            "chunks_used": 0,
            "tokens_used": 0,
            "latency_ms" : 0,
        }

    source_filter = _detect_source_filter(question.lower())
    result = _rag_agent.run(question, source_filter=source_filter)
    return result


def _sql_answer(question: str) -> dict:
    """Query SQLite for real metrics and return a natural language answer."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()

        cur.execute(
            "SELECT COUNT(*), AVG(fraud_score), MAX(fraud_score), "
            "SUM(CASE WHEN label=1 THEN 1 ELSE 0 END) "
            "FROM transactions WHERE fraud_score IS NOT NULL"
        )
        row = cur.fetchone()
        conn.close()

        total      = row[0] or 0
        avg_score  = round(row[1], 4) if row[1] else 0
        max_score  = round(row[2], 4) if row[2] else 0
        fraud_count= row[3] or 0

        answer = (
            f"From the transactions database: {total:,} transactions have been scored. "
            f"Average fraud score: {avg_score:.4f}. "
            f"Highest score: {max_score:.4f}. "
            f"Flagged as fraud (score ≥ 0.5): {fraud_count:,} transactions "
            f"({(fraud_count/total*100):.1f}% of total)."
            if total > 0 else
            "No scored transactions found in the database yet. "
            "Ingest and score some transactions via POST /ingest and POST /predict."
        )

        return {"answer": answer, "sources": ["finrisk.db"], "chunks_used": 0, "tokens_used": 0}

    except Exception as e:
        log.error(f"SQL agent DB query failed: {e}")
        return {
            "answer"     : f"Database query failed: {str(e)}",
            "sources"    : [],
            "chunks_used": 0,
            "tokens_used": 0,
        }


def _ml_answer(question: str) -> dict:
    """Return ML model metadata and performance context."""
    model_info = "XGBoost v1.0" if _model is not None else "Model not loaded"
    n_features = len(_feature_cols) if _feature_cols else 0

    answer = (
        f"The fraud detection model ({model_info}) uses {n_features} features. "
        "Key predictors are: amount_zscore (transaction amount relative to account history), "
        "velocity_24tx (total spend in last 24h rolling window), "
        "hour_of_day (transactions at 1–4am carry 3× higher fraud rate), "
        "and the V1–V28 PCA features from the card processor. "
        "A score ≥ 0.80 = HIGH risk (auto-block), 0.50–0.79 = MEDIUM (manual review), "
        "< 0.50 = LOW (cleared). "
        "Use POST /predict with a full feature vector to score any transaction in real time."
    )

    return {"answer": answer, "sources": ["fraud_model.pkl", "feature_names.txt"], "chunks_used": 0, "tokens_used": 0}

def _persist_agent_query(
    question: str,
    agent_used: str,
    answer: str,
    sources: list,
    chunks_used: int,
    tokens_used: int,
    latency_ms: int,
) -> None:
    """Store every agent query in SQLite for Power BI Agent Insights tab."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """INSERT INTO agent_queries
               (question, agent_used, answer, sources, chunks_used, tokens_used, latency_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                question,
                agent_used,
                answer,
                json.dumps(sources),
                chunks_used,
                tokens_used,
                latency_ms,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning(f"Could not persist agent query: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 1 — POST /ingest
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/ingest", tags=["Data"], summary="Ingest transaction or portfolio data")
def ingest(req: IngestRequest):
    """
    Validate and store a transaction or portfolio snapshot.
    Writes to SQLite (swap the sqlite3 calls for Azure SQL in Phase 5).
    """
    log.info(f"POST /ingest  data_type={req.data_type}")

    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()

        if req.data_type == "transaction":
            if not req.transaction:
                raise HTTPException(status_code=422, detail="'transaction' payload is required")
            tx = req.transaction
            cur.execute(
                """INSERT OR REPLACE INTO transactions
                   (tx_id, amount, merchant, timestamp, features)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    tx.tx_id,
                    tx.amount,
                    tx.merchant,
                    tx.timestamp,
                    json.dumps(tx.features) if tx.features else None,
                ),
            )
            conn.commit()
            log.info(f"Ingested transaction {tx.tx_id}  amount={tx.amount}")
            return {
                "status"    : "ok",
                "ingested"  : "transaction",
                "tx_id"     : tx.tx_id,
                "stored_at" : datetime.utcnow().isoformat(),
            }

        else:  # portfolio
            if not req.portfolio:
                raise HTTPException(status_code=422, detail="'portfolio' payload is required")
            pf = req.portfolio
            for h in pf.holdings:
                cur.execute(
                    """INSERT INTO portfolio_snapshots
                       (snapshot_date, total_value_inr, asset_class, weight)
                       VALUES (?, ?, ?, ?)""",
                    (pf.snapshot_date, pf.total_value_inr, h.get("asset", "unknown"), h.get("weight", 0.0)),
                )
            conn.commit()
            log.info(f"Ingested portfolio {pf.snapshot_date}  holdings={len(pf.holdings)}")
            return {
                "status"         : "ok",
                "ingested"       : "portfolio",
                "snapshot_date"  : pf.snapshot_date,
                "holdings_count" : len(pf.holdings),
                "stored_at"      : datetime.utcnow().isoformat(),
            }

    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail=f"Duplicate tx_id: {req.transaction.tx_id}")
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 2 — POST /predict
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/predict", tags=["ML"], summary="Score a transaction for fraud risk")
def predict(req: PredictRequest):
    """
    Run the XGBoost fraud classifier on a feature vector.
    Returns fraud_score (0–1), binary label, and risk tier.
    """
    log.info(f"POST /predict  tx_id={req.tx_id or 'anonymous'}")

    if _model is None or _feature_cols is None:
        raise HTTPException(status_code=503, detail="Model not loaded — check startup logs")

    try:
        feature_values = [getattr(req, col) for col in _feature_cols]
        X = np.array(feature_values, dtype=np.float64).reshape(1, -1)

        fraud_score = float(_model.predict_proba(X)[0][1])
        label       = int(fraud_score >= 0.5)

        if fraud_score >= 0.80:
            risk_level = "HIGH"
        elif fraud_score >= 0.50:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Write score back to DB if tx_id supplied
        if req.tx_id:
            try:
                conn = sqlite3.connect(DB_PATH)
                conn.execute(
                    "UPDATE transactions SET fraud_score=?, label=? WHERE tx_id=?",
                    (fraud_score, label, req.tx_id),
                )
                conn.commit()
                conn.close()
            except Exception as db_err:
                log.warning(f"DB score update failed: {db_err}")

        log.info(f"Prediction  score={fraud_score:.4f}  label={label}  risk={risk_level}")
        return {
            "tx_id"       : req.tx_id,
            "fraud_score" : round(fraud_score, 4),
            "label"       : label,
            "risk_level"  : risk_level,
            "model"       : "XGBoost v1.0",
            "predicted_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        log.error(f"Predict error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 3 — GET /search
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/search", tags=["Regulatory"], summary="Vector search over regulatory documents")
def search(
    q     : str           = Query(..., min_length=3, examples=["SEBI AIF Category II disclosure"]),
    source: Optional[str] = Query(None, examples=["SEBI"]),
    top_k : int           = Query(5, ge=1, le=20),
):
    """
    Embed the query with HuggingFace all-MiniLM-L6-v2, search the FAISS
    index, and return the top-k most relevant regulatory document chunks.
    Requires the index to have been built first:
        python agents/rag_pipeline.py index
    """
    log.info(f"GET /search  q='{q}'  source={source}  top_k={top_k}")

    # ── RAG not ready ──────────────────────────────────────────────────────────
    if _rag_agent is None or not _rag_agent._ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "Regulatory document index not built. "
                "Run: python agents/rag_pipeline.py index"
            ),
        )

    try:
        from agents.rag_pipeline import retrieve_chunks

        # Use detected source filter OR the explicit query param
        effective_source = source or _detect_source_filter(q.lower())
        chunks = retrieve_chunks(q, source_filter=effective_source)

        results = [
            {
                "rank"           : i + 1,
                "source"         : c["source"],
                "source_type"    : c["source_type"],
                "relevance_score": c["relevance_score"],
                "chunk"          : c["content"],
            }
            for i, c in enumerate(chunks[:top_k])
        ]

        return {
            "query"         : q,
            "source_filter" : effective_source,
            "total_results" : len(results),
            "results"       : results,
            "backend"       : "FAISS local vector search · all-MiniLM-L6-v2",
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 4 — POST /agent/query
# ══════════════════════════════════════════════════════════════════════════════

# Intent keyword lists
_RAG_KEYWORDS = {
    "sebi", "rbi", "regulation", "compliance", "basel", "fema",
    "guideline", "circular", "disclosure", "aif", "irdai", "nbfc",
    "capital adequacy", "cet1", "tier 1", "nsfr", "lcr", "insurance",
    "foreign exchange", "fdi", "monetary", "reserve bank", "securities",
}
_SQL_KEYWORDS = {
    "average", "count", "total", "how many", "last week", "last month",
    "sum", "fraud score", "transaction", "how much", "volume", "rate",
    "percentage", "trend", "history", "records", "database",
}
_ML_KEYWORDS = {
    "explain", "why", "score", "probability", "feature", "model",
    "predict", "anomaly", "shap", "importance", "threshold", "accuracy",
    "precision", "recall", "auc", "xgboost", "classifier",
}


def _route_question(question_lower: str) -> str:
    """Return which agent should handle this question."""
    if any(k in question_lower for k in _RAG_KEYWORDS):
        return "rag_agent"
    if any(k in question_lower for k in _SQL_KEYWORDS):
        return "sql_agent"
    if any(k in question_lower for k in _ML_KEYWORDS):
        return "ml_expert_agent"
    return "orchestrator"


@app.post("/agent/query", tags=["Agents"])
def agent_query(req: AgentQueryRequest):
    import time
    start = time.time()
    log.info(f"POST /agent/query — question='{req.question[:60]}'")

    question_lower = req.question.lower()

    sql_keywords = ["average", "count", "total", "how many", "last week",
                    "last month", "sum", "fraud score", "transaction"]
    rag_keywords = ["sebi", "rbi", "regulation", "compliance", "basel",
                    "fema", "guideline", "circular", "disclosure", "aif"]
    ml_keywords  = ["explain", "why", "score", "probability", "feature",
                    "model", "predict", "anomaly", "fraud"]

    extra_meta = {}

    # ── RAG agent ─────────────────────────────────────────────────────────────
    if any(k in question_lower for k in rag_keywords):
        agent_used    = "rag_agent"
        source_filter = next(
            (s for s in ("SEBI", "RBI", "Basel", "FEMA")
             if s.lower() in question_lower),
            None
        )
        result     = _rag_agent.run(req.question, source_filter=source_filter)
        answer     = result["answer"]
        extra_meta = {
            "sources"    : result.get("sources", []),
            "chunks_used": result.get("chunks_used", 0),
            "tokens_used": result.get("tokens_used", 0),
        }

    # ── Fraud expert agent ────────────────────────────────────────────────────
    elif any(k in question_lower for k in ml_keywords):
        agent_used = "fraud_expert_agent"

        # Try to pull a real fraud score + features from DB by tx_id
        tx_id, fraud_score, features = _extract_tx_context(req.question)

        result     = _fraud_agent.explain(
            fraud_score=fraud_score,
            features=features,
            tx_id=tx_id,
        )
        answer     = result["explanation"]
        extra_meta = {
            "fraud_score" : result.get("fraud_score"),
            "risk_level"  : result.get("risk_level"),
            "action"      : result.get("action"),
            "top_features": result.get("top_features", []),
            "thinking"    : result.get("thinking", ""),
            "tokens_used" : result.get("tokens_used", 0),
        }

    # ── SQL agent (stub — Phase 5) ────────────────────────────────────────────
    elif any(k in question_lower for k in sql_keywords):
        agent_used = "sql_agent"
        answer     = _stub_sql_answer(req.question)

    # ── Orchestrator fallback ─────────────────────────────────────────────────
    else:
        agent_used = "orchestrator"
        answer     = (
            f"I received: '{req.question}'. "
            "Please ask about regulations (RAG), fraud explanations (ML expert), "
            "or transaction statistics (SQL)."
        )

    latency_ms = int((time.time() - start) * 1000)

    # Store in DB for Power BI agent insights tab
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

    return {
        "question"  : req.question,
        "agent_used": agent_used,
        "answer"    : answer,
        "latency_ms": latency_ms,
        "session_id": req.session_id,
        **extra_meta,
    }
# ══════════════════════════════════════════════════════════════════════════════
# UTILITY ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════
def _extract_tx_context(question: str) -> tuple:
    """
    Try to extract tx_id from the question and look up its score + features from DB.
    Falls back to a sample high-risk transaction if nothing found.
    """
    import re
    import json

    # Look for TX-##### pattern in the question
    match = re.search(r"TX-\d+", question, re.IGNORECASE)
    tx_id = match.group(0) if match else None

    if tx_id:
        try:
            conn = sqlite3.connect(DB_PATH)
            row  = conn.execute(
                "SELECT fraud_score, features FROM transactions WHERE tx_id=?",
                (tx_id,)
            ).fetchone()
            conn.close()

            if row and row[0] is not None:
                fraud_score = row[0]
                features    = json.loads(row[1]) if row[1] else {}
                log.info(f"Found tx {tx_id} in DB — score={fraud_score:.4f}")
                return tx_id, fraud_score, features
        except Exception as e:
            log.warning(f"DB lookup failed for {tx_id}: {e}")

    # Fallback — sample high-risk values for demo purposes
    log.info("No tx found in DB — using sample high-risk features")
    return tx_id, 0.94, {
        "amount_zscore": 4.1, "velocity_24tx": 8200.0,
        "hour_of_day": 3,     "amount_log": 9.2,
        "V1": -3.2, "V2": 2.8, "V3": -2.1, "V4": 1.5,
        "V5": -3.8, "V14": -4.1, "V17": -3.4,
    }
@app.get("/health", tags=["System"], summary="System health check")
def health():
    """Returns status of model, RAG index, and database."""
    rag_status = "not_loaded"
    if _rag_agent is not None:
        rag_status = "ready" if _rag_agent._ready else "index_not_built"

    return {
        "status"        : "ok",
        "model_loaded"  : _model is not None,
        "features"      : len(_feature_cols) if _feature_cols else 0,
        "rag_status"    : rag_status,
        "db_path"       : str(DB_PATH),
        "timestamp"     : datetime.utcnow().isoformat(),
    }


@app.get("/forecast/portfolio", tags=["ML"], summary="Prophet portfolio forecast")
def get_portfolio_forecast(days: int = Query(30, ge=7, le=90)):
    """
    Return historical portfolio values + forward forecast from the Prophet model.
    Requires ml/train_forecaster.py to have been run first.
    """
    try:
        df = pd.read_csv(FORECAST_PATH)
        df["ds"] = pd.to_datetime(df["ds"])

        today   = df["ds"].max()
        cutoff  = today - pd.Timedelta(days=60)
        out_df  = df[df["ds"] >= cutoff].copy()
        out_df["ds"] = out_df["ds"].dt.strftime("%Y-%m-%d")

        return {
            "forecast_days": days,
            "rows"         : len(out_df),
            "data"         : out_df.to_dict(orient="records"),
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Portfolio forecast CSV not found. Run ml/train_forecaster.py first.",
        )
    except Exception as e:
        log.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/history", tags=["Agents"], summary="Recent agent query history")
def agent_history(limit: int = Query(20, ge=1, le=100)):
    """
    Return the last N agent queries from SQLite.
    Used by the Power BI Agent Insights tab and the frontend Agent Console.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute(
            """SELECT question, agent_used, answer, sources, chunks_used,
                      tokens_used, latency_ms, created_at
               FROM agent_queries
               ORDER BY id DESC
               LIMIT ?""",
            (limit,),
        )
        rows = cur.fetchall()
        conn.close()

        return {
            "count": len(rows),
            "history": [
                {
                    "question"   : r[0],
                    "agent_used" : r[1],
                    "answer"     : r[2],
                    "sources"    : json.loads(r[3]) if r[3] else [],
                    "chunks_used": r[4],
                    "tokens_used": r[5],
                    "latency_ms" : r[6],
                    "created_at" : r[7],
                }
                for r in rows
            ],
        }

    except Exception as e:
        log.error(f"History fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
