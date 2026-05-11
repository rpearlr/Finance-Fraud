import time
from fastapi import APIRouter, HTTPException, Query
from backend.schemas import AgentQueryRequest
from backend.database import query_db
from backend.config import log
from backend.utils import _persist_agent_query, _extract_tx_context
from backend.state import get_state

router = APIRouter(tags=["Agents"])

@router.post("/agent/query")
def agent_query(req: AgentQueryRequest):
    start        = time.time()
    state        = get_state()
    orchestrator = state["orchestrator"]

    if not orchestrator:
        log.error("Orchestrator not initialized")
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")

    # The orchestrator now handles the entire execution loop via LangChain AgentExecutor
    result = orchestrator.run(req.question)
    
    answer     = result["answer"]
    agent_used = result["agent_used"]
    extra_meta = result["extra_meta"]

    latency_ms = int((time.time() - start) * 1000)
    
    # Persist the query to the database
    _persist_agent_query(
        question=req.question, 
        agent_used=agent_used, 
        answer=answer,
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

@router.get("/agent/history", summary="Recent agent query history")
def agent_history(limit: int = Query(20, ge=1, le=100)):
    try:
        rows = query_db(
            "SELECT * FROM agent_queries ORDER BY created_at DESC LIMIT ?", 
            (limit,)
        )
        return {"count": len(rows), "history": rows}
    except Exception as e:
        log.error(f"History fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
