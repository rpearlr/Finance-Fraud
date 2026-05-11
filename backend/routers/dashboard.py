import math
from fastapi import APIRouter, HTTPException
from backend.database import query_db, query_one
from backend.config import log

router = APIRouter(tags=["Dashboard"])

@router.get("/dashboard/kpis", summary="Get overview dashboard metrics")
def dashboard_kpis():
    try:
        # Basic stats
        total = query_one("SELECT COUNT(*) as c FROM transactions")["c"]
        fraud_count = query_one("SELECT COUNT(*) as c FROM transactions WHERE label = 1")["c"]
        
        # Average fraud score
        avg_res = query_one("SELECT AVG(fraud_score) as avg FROM transactions WHERE fraud_score IS NOT NULL")
        raw_avg = avg_res["avg"] if avg_res and avg_res["avg"] is not None else 0.0
        avg_score = 0.0 if math.isnan(raw_avg) else round(raw_avg, 3)

        # Latest portfolio value
        pf_row = query_one("SELECT total_value_inr FROM portfolio_snapshots ORDER BY snapshot_date DESC LIMIT 1")
        pf_value = pf_row["total_value_inr"] if pf_row else 0.0

        # Hourly fraud aggregation using SQL strftime
        hourly_raw = query_db("""
            SELECT 
                strftime('%H', timestamp) as hour,
                AVG(fraud_score) as avg_score
            FROM transactions 
            WHERE fraud_score IS NOT NULL
            GROUP BY hour
            ORDER BY hour ASC
        """)
        hourly_fraud = []
        for r in hourly_raw:
            val = r["avg_score"] if r["avg_score"] is not None else 0.0
            val = 0.0 if math.isnan(val) else round(val, 4)
            hourly_fraud.append({"hour": str(r["hour"]).zfill(2), "avg_score": val})

        # Recent fraud alerts
        alert_rows = query_db("""
            SELECT tx_id, amount, fraud_score, timestamp 
            FROM transactions 
            WHERE label = 1 
            ORDER BY timestamp DESC 
            LIMIT 5
        """)
        recent_alerts = [
            {
                "tx_id" : r["tx_id"],
                "amount": r["amount"],
                "score" : round(r["fraud_score"], 2),
                "time"  : r["timestamp"],
            }
            for r in alert_rows
        ]

        # Active agents count
        agent_res = query_one("SELECT COUNT(DISTINCT agent_used) as c FROM agent_queries")
        active_agents = agent_res["c"] if agent_res and agent_res["c"] else 3

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
