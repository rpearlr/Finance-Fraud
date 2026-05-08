from fastapi import APIRouter, HTTPException
from backend.database import col
from backend.config import log

router = APIRouter(tags=["Dashboard"])

@router.get("/dashboard/kpis", summary="Get overview dashboard metrics")
def dashboard_kpis():
    try:
        txs         = col("transactions")
        total       = txs.count_documents({})
        fraud_count = txs.count_documents({"label": 1})

        agg = txs.aggregate([
            {"$match": {"fraud_score": {"$exists": True}}},
            {"$group": {"_id": None, "avg": {"$avg": "$fraud_score"}}},
        ])
        avg_score = round(agg[0]["avg"], 3) if agg else 0.0

        pf_row   = col("portfolio_snapshots").find_one(sort=[("snapshot_date", -1)])
        pf_value = pf_row["total_value_inr"] if pf_row else 0.0

        # Hourly fraud aggregation
        hourly_raw = txs.aggregate([
            {"$match": {"fraud_score": {"$exists": True}}},
            {"$group": {
                "_id"      : {"$hour": "$timestamp"},
                "avg_score": {"$avg": "$fraud_score"},
            }},
            {"$sort": {"_id": 1}},
        ])
        hourly_fraud = [
            {"hour": str(r["_id"]).zfill(2), "avg_score": round(r["avg_score"], 4)}
            for r in hourly_raw
        ]

        # Recent fraud alerts
        alert_rows    = txs.find(
            {"label": 1},
            {"tx_id": 1, "amount": 1, "fraud_score": 1, "timestamp": 1},
            sort=[("timestamp", -1)],
            limit=5,
        )
        recent_alerts = [
            {
                "tx_id" : r["tx_id"],
                "amount": r["amount"],
                "score" : round(r["fraud_score"], 2),
                "time"  : r["timestamp"] if isinstance(r["timestamp"], str) else str(r["timestamp"]),
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
