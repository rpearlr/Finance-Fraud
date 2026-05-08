from datetime import datetime
from fastapi import APIRouter
from backend.database import get_conn
from backend.config import SQLITE_PATH
from backend.state import get_state

router = APIRouter(tags=["System"])

@router.get("/health", summary="System health check")
def health():
    state = get_state()
    model = state["model"]
    feature_cols = state["feature_cols"]
    rag_agent = state["rag_agent"]

    rag_status = "not_loaded"
    if rag_agent is not None:
        rag_status = "ready" if rag_agent._ready else "index_not_built"

    db_status = "unknown"
    try:
        get_conn().execute("SELECT 1")
        db_status = "connected"
    except Exception:
        db_status = "unreachable"

    return {
        "status"      : "ok",
        "model_loaded": model is not None,
        "features"    : len(feature_cols) if feature_cols else 0,
        "rag_status"  : rag_status,
        "db_status"   : db_status,
        "db_path"     : SQLITE_PATH,
        "timestamp"   : datetime.utcnow().isoformat(),
    }
