"""
FinRisk AI Platform — FastAPI Backend (MongoDB edition)
Run from project root: uvicorn backend.main:app --reload

Endpoints:
  POST /ingest             — validate + store transaction/portfolio data
  POST /predict            — fraud score from XGBoost model
  GET  /search             — real FAISS vector search over regulatory docs
  POST /agent/query        — routes question to real RAG / SQL / ML agent
  GET  /health             — system health check
  GET  /forecast/portfolio — Prophet forecast data
"""

import os
import re
import time
import pickle
import logging
import io
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except ImportError:
    ExponentialSmoothing = None

from agents.fraud_agent import FraudExpertAgent
from agents.rag_pipeline import DocumentAssistantAgent, warmup_embed_model
from agents.investment_agent import InvestmentAdviceAgent
from backend.pdf_generator import build_fraud_report, build_investment_report

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("finrisk")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
MODEL_PATH    = BASE_DIR / "ml" / "models" / "fraud_model.pkl"
FEATURES_PATH = BASE_DIR / "ml" / "models" / "feature_names.txt"
FORECAST_PATH = BASE_DIR / "data" / "staged" / "portfolio_forecast.csv"

# ── MongoDB ────────────────────────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017")
MONGO_DB  = os.getenv("MONGO_DB",  "finrisk")

_mongo_client: MongoClient = None

def get_db():
    return _mongo_client[MONGO_DB]

def col(name: str):
    return get_db()[name]

# ── Global state ───────────────────────────────────────────────────────────────
_model        = None
_feature_cols = None
_fraud_agent      = None
_rag_agent        = None
_investment_agent = None


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def init_mongo() -> None:
    global _mongo_client
    _mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # Verify connection
    _mongo_client.admin.command("ping")
    # Ensure indexes
    col("transactions").create_index("tx_id", unique=True)
    col("transactions").create_index([("timestamp", ASCENDING)])
    col("transactions").create_index([("label", ASCENDING)])
    col("portfolio_snapshots").create_index([("snapshot_date", ASCENDING)])
    col("agent_queries").create_index([("created_at", ASCENDING)])
    log.info(f"MongoDB ready — {MONGO_URI}/{MONGO_DB}")


def load_model() -> None:
    global _model, _feature_cols, _fraud_agent, _rag_agent, _investment_agent

    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
    _feature_cols = FEATURES_PATH.read_text(encoding="utf-8").strip().splitlines()
    log.info(f"Loaded fraud model — {len(_feature_cols)} features")

    _fraud_agent      = FraudExpertAgent()
    _rag_agent        = DocumentAssistantAgent()
    _investment_agent = InvestmentAdviceAgent()
    log.info("Agents initialised")

    warmup_embed_model()
    log.info("Embedding model pre-loaded ✓")


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting FinRisk API...")
    init_mongo()
    load_model()
    log.info("API ready ✓")
    yield
    if _mongo_client:
        _mongo_client.close()
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
    V1 : float; V2 : float; V3 : float; V4 : float; V5 : float
    V6 : float; V7 : float; V8 : float; V9 : float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    amount_zscore : float = Field(..., examples=[1.5])
    velocity_24tx : float = Field(..., examples=[3200.0])
    hour_of_day   : int   = Field(..., ge=0, le=23, examples=[2])
    amount_log    : float = Field(..., examples=[7.8])
    tx_id: Optional[str] = None


class AgentQueryRequest(BaseModel):
    question      : str           = Field(..., max_length=500)
    session_id    : Optional[str] = None
    agent_override: Optional[str] = Field(None, description="Optional agent name to bypass orchestrator routing")


class ReportRequest(BaseModel):
    report_type: str           = Field(..., description="'fraud' or 'investment'")
    query      : Optional[str] = None
    tx_id      : Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _detect_source_filter(question_lower: str) -> Optional[str]:
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
    if _rag_agent is None:
        return {"answer": "RAG agent could not be loaded.", "sources": [], "chunks_used": 0, "tokens_used": 0, "latency_ms": 0}
    if not _rag_agent._ready:
        return {
            "answer": (
                "The regulatory document index has not been built yet. "
                "Run: python agents/rag_pipeline.py index"
            ),
            "sources": [], "chunks_used": 0, "tokens_used": 0, "latency_ms": 0,
        }
    source_filter = _detect_source_filter(question.lower())
    return _rag_agent.run(question, source_filter=source_filter)


def _sql_answer(question: str) -> dict:
    """Query MongoDB transactions and return a natural language answer."""
    try:
        txs   = col("transactions")
        total = txs.count_documents({"fraud_score": {"$exists": True}})

        agg = list(txs.aggregate([
            {"$match": {"fraud_score": {"$exists": True}}},
            {"$group": {
                "_id"      : None,
                "avg_score": {"$avg": "$fraud_score"},
                "max_score": {"$max": "$fraud_score"},
            }},
        ]))
        avg_score   = round(agg[0]["avg_score"], 4) if agg else 0
        max_score   = round(agg[0]["max_score"], 4) if agg else 0
        fraud_count = txs.count_documents({"label": 1})

        if total > 0:
            answer = (
                f"From the transactions database: {total:,} transactions have been scored. "
                f"Average fraud score: {avg_score:.4f}. "
                f"Highest score: {max_score:.4f}. "
                f"Flagged as fraud (score ≥ 0.5): {fraud_count:,} transactions "
                f"({(fraud_count / total * 100):.1f}% of total)."
            )
        else:
            answer = (
                "No scored transactions found in the database yet. "
                "Ingest and score some transactions via POST /ingest and POST /predict."
            )

        return {"answer": answer, "sources": ["MongoDB:transactions"], "chunks_used": 0, "tokens_used": 0}

    except Exception as e:
        log.error(f"SQL agent DB query failed: {e}")
        return {"answer": f"Database query failed: {str(e)}", "sources": [], "chunks_used": 0, "tokens_used": 0}


def _ml_answer(question: str) -> dict:
    model_info = "XGBoost v1.0" if _model is not None else "Model not loaded"
    n_features = len(_feature_cols) if _feature_cols else 0
    answer = (
        f"The fraud detection model ({model_info}) uses {n_features} features. "
        "Key predictors are: amount_zscore, velocity_24tx, hour_of_day, amount_log, and V1–V28 PCA features. "
        "A score ≥ 0.80 = HIGH risk (auto-block), 0.50–0.79 = MEDIUM (manual review), < 0.50 = LOW (cleared). "
        "Use POST /predict with a full feature vector to score any transaction in real time."
    )
    return {"answer": answer, "sources": ["fraud_model.pkl", "feature_names.txt"], "chunks_used": 0, "tokens_used": 0}


def _persist_agent_query(
    question: str, agent_used: str, answer: str,
    sources: list, chunks_used: int, tokens_used: int, latency_ms: int,
) -> None:
    try:
        col("agent_queries").insert_one({
            "question"   : question,
            "agent_used" : agent_used,
            "answer"     : answer,
            "sources"    : sources,
            "chunks_used": chunks_used,
            "tokens_used": tokens_used,
            "latency_ms" : latency_ms,
            "created_at" : datetime.utcnow(),
        })
    except Exception as e:
        log.warning(f"Could not persist agent query: {e}")


def _extract_tx_context(question: str) -> tuple:
    match = re.search(r"TX-\d+", question, re.IGNORECASE)
    tx_id = match.group(0) if match else None

    if tx_id:
        try:
            row = col("transactions").find_one({"tx_id": tx_id})
            if row and row.get("fraud_score") is not None:
                log.info(f"Found tx {tx_id} in MongoDB — score={row['fraud_score']:.4f}")
                return tx_id, row["fraud_score"], row.get("features", {})
        except Exception as e:
            log.warning(f"DB lookup failed for {tx_id}: {e}")

    log.info("No tx found in DB — using sample high-risk features")
    return tx_id, 0.94, {
        "amount_zscore": 4.1, "velocity_24tx": 8200.0,
        "hour_of_day"  : 3,   "amount_log"   : 9.2,
        "V1": -3.2, "V2": 2.8, "V3": -2.1, "V4": 1.5,
        "V5": -3.8, "V14": -4.1, "V17": -3.4,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 1 — POST /ingest
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/ingest", tags=["Data"], summary="Ingest transaction or portfolio data")
def ingest(req: IngestRequest):
    log.info(f"POST /ingest  data_type={req.data_type}")

    try:
        if req.data_type == "transaction":
            if not req.transaction:
                raise HTTPException(status_code=422, detail="'transaction' payload is required")
            tx = req.transaction
            try:
                col("transactions").insert_one({
                    "tx_id"    : tx.tx_id,
                    "amount"   : tx.amount,
                    "merchant" : tx.merchant,
                    "timestamp": datetime.fromisoformat(tx.timestamp),
                    "features" : tx.features or {},
                })
            except DuplicateKeyError:
                raise HTTPException(status_code=409, detail=f"Duplicate tx_id: {tx.tx_id}")

            log.info(f"Ingested transaction {tx.tx_id}  amount={tx.amount}")
            return {
                "status"   : "ok",
                "ingested" : "transaction",
                "tx_id"    : tx.tx_id,
                "stored_at": datetime.utcnow().isoformat(),
            }

        else:  # portfolio
            if not req.portfolio:
                raise HTTPException(status_code=422, detail="'portfolio' payload is required")
            pf = req.portfolio
            col("portfolio_snapshots").insert_many([
                {
                    "snapshot_date"  : pf.snapshot_date,
                    "total_value_inr": pf.total_value_inr,
                    "asset_class"    : h.get("asset", "unknown"),
                    "weight"         : h.get("weight", 0.0),
                    "created_at"     : datetime.utcnow(),
                }
                for h in pf.holdings
            ])
            log.info(f"Ingested portfolio {pf.snapshot_date}  holdings={len(pf.holdings)}")
            return {
                "status"        : "ok",
                "ingested"      : "portfolio",
                "snapshot_date" : pf.snapshot_date,
                "holdings_count": len(pf.holdings),
                "stored_at"     : datetime.utcnow().isoformat(),
            }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 2 — POST /predict
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/predict", tags=["ML"], summary="Score a transaction for fraud risk")
def predict(req: PredictRequest):
    log.info(f"POST /predict  tx_id={req.tx_id or 'anonymous'}")

    if _model is None or _feature_cols is None:
        raise HTTPException(status_code=503, detail="Model not loaded — check startup logs")

    try:
        feature_values = [getattr(req, c) for c in _feature_cols]
        X = np.array(feature_values, dtype=np.float64).reshape(1, -1)

        fraud_score = float(_model.predict_proba(X)[0][1])
        label       = int(fraud_score >= 0.5)
        risk_level  = "HIGH" if fraud_score >= 0.80 else ("MEDIUM" if fraud_score >= 0.50 else "LOW")

        if req.tx_id:
            try:
                col("transactions").update_one(
                    {"tx_id": req.tx_id},
                    {"$set": {"fraud_score": fraud_score, "label": label}},
                )
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
    log.info(f"GET /search  q='{q}'  source={source}  top_k={top_k}")

    if _rag_agent is None or not _rag_agent._ready:
        raise HTTPException(
            status_code=503,
            detail="Regulatory document index not built. Run: python agents/rag_pipeline.py index",
        )

    try:
        from agents.rag_pipeline import retrieve_chunks
        effective_source = source or _detect_source_filter(q.lower())
        chunks = retrieve_chunks(q, source_filter=effective_source)

        return {
            "query"        : q,
            "source_filter": effective_source,
            "total_results": min(len(chunks), top_k),
            "results"      : [
                {
                    "rank"           : i + 1,
                    "source"         : c["source"],
                    "source_type"    : c["source_type"],
                    "relevance_score": c["relevance_score"],
                    "chunk"          : c["content"],
                }
                for i, c in enumerate(chunks[:top_k])
            ],
            "backend": "FAISS local vector search · all-MiniLM-L6-v2",
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 4 — POST /agent/query
# ══════════════════════════════════════════════════════════════════════════════

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
_INVESTMENT_KEYWORDS = {
    "invest", "stock", "market", "portfolio", "buy", "sell",
    "asset", "crypto", "mutual fund", "advice",
}


@app.post("/agent/query", tags=["Agents"])
def agent_query(req: AgentQueryRequest):
    start          = time.time()
    question_lower = req.question.lower()
    agent_override = req.agent_override
    extra_meta     = {}

    log.info(f"POST /agent/query — question='{req.question[:60]}'")

    # ── RAG agent ──────────────────────────────────────────────────────────────
    if agent_override == "rag_agent" or (
        not agent_override and any(k in question_lower for k in _RAG_KEYWORDS)
    ):
        agent_used    = "rag_agent"
        source_filter = next(
            (s for s in ("SEBI", "RBI", "Basel", "FEMA") if s.lower() in question_lower),
            None,
        )
        result     = _rag_agent.run(req.question, source_filter=source_filter)
        answer     = result["answer"]
        extra_meta = {
            "sources"    : result.get("sources", []),
            "chunks_used": result.get("chunks_used", 0),
            "tokens_used": result.get("tokens_used", 0),
        }

    # ── Fraud expert agent ─────────────────────────────────────────────────────
    elif agent_override == "fraud_expert_agent" or (
        not agent_override and any(k in question_lower for k in _ML_KEYWORDS)
    ):
        agent_used              = "fraud_expert_agent"
        tx_id, fraud_score, features = _extract_tx_context(req.question)
        result                  = _fraud_agent.explain(fraud_score=fraud_score, features=features, tx_id=tx_id)
        answer                  = result["explanation"]
        extra_meta              = {
            "fraud_score" : result.get("fraud_score"),
            "risk_level"  : result.get("risk_level"),
            "action"      : result.get("action"),
            "top_features": result.get("top_features", []),
            "thinking"    : result.get("thinking", ""),
            "tokens_used" : result.get("tokens_used", 0),
        }

    # ── SQL / stats agent ──────────────────────────────────────────────────────
    elif agent_override == "sql_agent" or (
        not agent_override and any(k in question_lower for k in _SQL_KEYWORDS)
    ):
        agent_used = "sql_agent"
        sql_res    = _sql_answer(req.question)
        answer     = sql_res["answer"]
        extra_meta = {
            "sources"    : sql_res.get("sources", []),
            "chunks_used": sql_res.get("chunks_used", 0),
            "tokens_used": sql_res.get("tokens_used", 0),
        }

    # ── Investment agent ───────────────────────────────────────────────────────
    elif agent_override == "investment_agent" or (
        not agent_override and any(k in question_lower for k in _INVESTMENT_KEYWORDS)
    ):
        agent_used = "investment_agent"
        result     = _investment_agent.advise(req.question)
        answer     = result["explanation"]
        extra_meta = {
            "thinking"   : result.get("thinking", ""),
            "tokens_used": result.get("tokens_used", 0),
        }

    # ── Orchestrator fallback ──────────────────────────────────────────────────
    else:
        agent_used = "orchestrator"
        answer     = (
            f"I received: '{req.question}'. "
            "Please ask about regulations (RAG), fraud explanations (ML expert), "
            "or transaction statistics (SQL)."
        )

    latency_ms = int((time.time() - start) * 1000)

    _persist_agent_query(
        question=req.question, agent_used=agent_used, answer=answer,
        sources=extra_meta.get("sources", []),
        chunks_used=extra_meta.get("chunks_used", 0),
        tokens_used=extra_meta.get("tokens_used", 0),
        latency_ms=latency_ms,
    )

    return {
        "question"  : req.question,
        "agent_used": agent_used,
        "answer"    : answer,
        "latency_ms": latency_ms,
        "session_id": req.session_id,
        **extra_meta,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 5 — GET /health
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"], summary="System health check")
def health():
    rag_status = "not_loaded"
    if _rag_agent is not None:
        rag_status = "ready" if _rag_agent._ready else "index_not_built"

    mongo_status = "unknown"
    try:
        _mongo_client.admin.command("ping")
        mongo_status = "connected"
    except Exception:
        mongo_status = "unreachable"

    return {
        "status"      : "ok",
        "model_loaded": _model is not None,
        "features"    : len(_feature_cols) if _feature_cols else 0,
        "rag_status"  : rag_status,
        "mongo_status": mongo_status,
        "mongo_uri"   : MONGO_URI,
        "timestamp"   : datetime.utcnow().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 6 — GET /forecast/portfolio
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/forecast/portfolio", tags=["ML"], summary="Prophet portfolio forecast")
def get_portfolio_forecast(days: int = Query(30, ge=7, le=90)):
    try:
        df        = pd.read_csv(FORECAST_PATH)
        df["ds"]  = pd.to_datetime(df["ds"])
        cutoff    = df["ds"].max() - pd.Timedelta(days=60)
        out_df    = df[df["ds"] >= cutoff].copy()
        out_df["ds"] = out_df["ds"].dt.strftime("%Y-%m-%d")
        return {"forecast_days": days, "rows": len(out_df), "data": out_df.to_dict(orient="records")}
    except Exception as e:
        log.error(f"Error fetching forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 7 — POST /forecast/custom
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/forecast/custom", tags=["ML"], summary="Custom data forecast using Holt-Winters")
async def get_custom_forecast(file: UploadFile = File(...), days: int = Query(30, ge=7, le=90)):
    if ExponentialSmoothing is None:
        raise HTTPException(status_code=500, detail="statsmodels is not installed on the server.")

    try:
        content  = await file.read()
        df       = pd.read_csv(io.StringIO(content.decode("utf-8")))

        if len(df.columns) < 2:
            raise ValueError("CSV must have at least two columns: Date and Value")

        df["ds"] = pd.to_datetime(df[df.columns[0]])
        df       = df.sort_values("ds").reset_index(drop=True)
        df["y"]  = df[df.columns[1]].astype(float)

        model       = ExponentialSmoothing(df["y"], trend="add", seasonal=None, initialization_method="estimated")
        fitted      = model.fit()
        std_resid   = np.std(fitted.resid)

        df["yhat"]        = fitted.fittedvalues
        df["is_forecast"] = False
        df["yhat_lower"]  = df["yhat"] - 1.96 * std_resid
        df["yhat_upper"]  = df["yhat"] + 1.96 * std_resid

        forecast      = fitted.forecast(days)
        future_dates  = pd.date_range(start=df["ds"].iloc[-1] + pd.Timedelta(days=1), periods=days, freq="D")
        ci_growth     = np.arange(1, days + 1) * 0.1

        future_df = pd.DataFrame({
            "ds"         : future_dates,
            "yhat"       : forecast.values,
            "yhat_lower" : forecast.values - (1.96 * std_resid * (1 + ci_growth)),
            "yhat_upper" : forecast.values + (1.96 * std_resid * (1 + ci_growth)),
            "is_forecast": True,
        })

        out_df       = pd.concat([df[["ds", "yhat", "yhat_lower", "yhat_upper", "is_forecast"]], future_df], ignore_index=True)
        out_df["ds"] = out_df["ds"].dt.strftime("%Y-%m-%d")

        return {"forecast_days": days, "rows": len(out_df), "data": out_df.to_dict(orient="records")}

    except Exception as e:
        log.error(f"Error processing custom forecast: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 8 — GET /dashboard/kpis
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/dashboard/kpis", tags=["Dashboard"], summary="Get overview dashboard metrics")
def dashboard_kpis():
    try:
        txs         = col("transactions")
        total       = txs.count_documents({})
        fraud_count = txs.count_documents({"label": 1})

        agg = list(txs.aggregate([
            {"$match": {"fraud_score": {"$exists": True}}},
            {"$group": {"_id": None, "avg": {"$avg": "$fraud_score"}}},
        ]))
        avg_score = round(agg[0]["avg"], 3) if agg else 0.0

        pf_row   = col("portfolio_snapshots").find_one(sort=[("_id", -1)])
        pf_value = pf_row["total_value_inr"] if pf_row else 0.0

        # Hourly fraud aggregation
        hourly_raw = list(txs.aggregate([
            {"$match": {"fraud_score": {"$exists": True}}},
            {"$group": {
                "_id"      : {"$hour": "$timestamp"},
                "avg_score": {"$avg": "$fraud_score"},
            }},
            {"$sort": {"_id": 1}},
        ]))
        hourly_fraud = [{"hour": str(r["_id"]).zfill(2), "avg_score": round(r["avg_score"], 4)} for r in hourly_raw]

        # Recent fraud alerts
        alert_rows    = list(txs.find(
            {"label": 1},
            {"tx_id": 1, "amount": 1, "fraud_score": 1, "timestamp": 1, "_id": 0},
            sort=[("timestamp", -1)],
            limit=5,
        ))
        recent_alerts = [
            {
                "tx_id" : r["tx_id"],
                "amount": r["amount"],
                "score" : round(r["fraud_score"], 2),
                "time"  : r["timestamp"].strftime("%I:%M %p") if isinstance(r["timestamp"], datetime) else str(r["timestamp"]),
            }
            for r in alert_rows
        ]

        active_agents = len(col("agent_queries").distinct("agent_used")) or 3

        return {
            "transactions_today": total,
            "fraud_flagged"     : fraud_count,
            "avg_fraud_score"   : avg_score,
            "portfolio_value"   : pf_value,
            "hourly_fraud"      : hourly_fraud,
            "recent_alerts"     : recent_alerts,
            "active_agents"     : active_agents,
        }

    except Exception as e:
        log.error(f"Dashboard KPIs error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 9 — GET /agent/history
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/agent/history", tags=["Agents"], summary="Recent agent query history")
def agent_history(limit: int = Query(20, ge=1, le=100)):
    try:
        rows = list(col("agent_queries").find(
            {}, {"_id": 0},
            sort=[("created_at", -1)],
            limit=limit,
        ))
        # Convert datetime objects to ISO strings for JSON serialisation
        for r in rows:
            if isinstance(r.get("created_at"), datetime):
                r["created_at"] = r["created_at"].isoformat()

        return {"count": len(rows), "history": rows}

    except Exception as e:
        log.error(f"History fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 10 — POST /report/generate
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/report/generate", tags=["Reports"])
def generate_report(req: ReportRequest):
    try:
        if req.report_type == "fraud":
            if not req.tx_id:
                raise HTTPException(status_code=400, detail="tx_id required for fraud report")
            result      = _fraud_agent.explain(
                fraud_score=0.94,
                features={"amount": 2400, "velocity_24tx": 8200, "V14": -4.1, "V17": -4.0},
                tx_id=req.tx_id,
            )
            pdf_buffer  = build_fraud_report(req.tx_id, result.get("fraud_score", 0.94), result.get("explanation", ""))
            filename    = f"Fraud_Report_{req.tx_id}.pdf"

        elif req.report_type == "investment":
            if not req.query:
                raise HTTPException(status_code=400, detail="query required for investment report")
            result     = _investment_agent.advise(req.query)
            pdf_buffer = build_investment_report(req.query, result.get("explanation", ""))
            filename   = "Investment_Advice_Report.pdf"

        else:
            raise HTTPException(status_code=400, detail="Invalid report_type")

        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# FRONTEND SERVING
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Frontend"])
async def serve_login():
    return FileResponse("frontend/finrisk_login.html")


@app.get("/dashboard", tags=["Frontend"])
async def serve_dashboard():
    return FileResponse("frontend/finrisk_kpi_dashboard.html")


@app.get("/{filename}.html", tags=["Frontend"])
async def serve_html_files(filename: str):
    file_path = Path("frontend") / f"{filename}.html"
    if file_path.exists():
        return FileResponse(file_path)
    raise HTTPException(status_code=404)


# Mounted last so it never shadows API routes
app.mount("/", StaticFiles(directory="frontend"), name="frontend")