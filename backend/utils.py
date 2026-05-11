import re
from datetime import datetime
from typing import Optional
from backend.config import log
from backend.database import execute_db, query_one

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

def _persist_agent_query(
    question: str, agent_used: str, answer: str,
    sources: list, chunks_used: int, tokens_used: int, latency_ms: int,
) -> None:
    try:
        execute_db(
            """INSERT INTO agent_queries 
               (question, agent_used, answer, sources, chunks_used, tokens_used, latency_ms) 
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (question, agent_used, answer, sources, chunks_used, tokens_used, latency_ms)
        )
    except Exception as e:
        log.warning(f"Could not persist agent query: {e}")

def _extract_tx_context(question: str) -> tuple:
    match = re.search(r"TX-\d+", question, re.IGNORECASE)
    tx_id = match.group(0) if match else None

    if tx_id:
        try:
            row = query_one("SELECT * FROM transactions WHERE tx_id = ?", (tx_id,))
            if row and row.get("fraud_score") is not None:
                log.info(f"Found tx {tx_id} in SQLite — score={row['fraud_score']:.4f}")
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
