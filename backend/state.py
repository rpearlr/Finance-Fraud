import pickle
from backend.config import MODEL_PATH, FEATURES_PATH, log
from backend.database import col
from agents.fraud_agent import FraudExpertAgent
from agents.rag_pipeline import DocumentAssistantAgent, warmup_embed_model
from agents.investment_agent import InvestmentAdviceAgent
from agents.orchestrator import LangChainOrchestrator, set_db_col_func

_model        = None
_feature_cols = None
_fraud_agent      = None
_rag_agent        = None
_investment_agent = None
_orchestrator      = None

def load_model() -> None:
    global _model, _feature_cols, _fraud_agent, _rag_agent, _investment_agent, _orchestrator

    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
    _feature_cols = FEATURES_PATH.read_text(encoding="utf-8").strip().splitlines()
    log.info(f"Loaded fraud model — {len(_feature_cols)} features")

    _fraud_agent      = FraudExpertAgent()
    _rag_agent        = DocumentAssistantAgent()
    _investment_agent = InvestmentAdviceAgent()

    set_db_col_func(col)
    _orchestrator = LangChainOrchestrator()

    log.info("Agents initialised")

    warmup_embed_model()
    log.info("Embedding model pre-loaded ✓")

def get_state():
    return {
        "model": _model,
        "feature_cols": _feature_cols,
        "fraud_agent": _fraud_agent,
        "rag_agent": _rag_agent,
        "investment_agent": _investment_agent,
        "orchestrator": _orchestrator
    }
