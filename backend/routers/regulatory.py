from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from backend.config import log
from backend.utils import _detect_source_filter
from backend.state import get_state

router = APIRouter(tags=["Regulatory"])

@router.get("/search", summary="Vector search over regulatory documents")
def search(
    q     : str           = Query(..., min_length=3, examples=["SEBI AIF Category II disclosure"]),
    source: Optional[str] = Query(None, examples=["SEBI"]),
    top_k : int           = Query(5, ge=1, le=20),
):
    log.info(f"GET /search  q='{q}'  source={source}  top_k={top_k}")
    state = get_state()
    rag_agent = state["rag_agent"]

    if rag_agent is None or not rag_agent._ready:
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
