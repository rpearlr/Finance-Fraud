import sqlite3
from datetime import datetime
from fastapi import APIRouter, HTTPException
from backend.schemas import IngestRequest
from backend.database import col
from backend.config import log

router = APIRouter(tags=["Data"])

@router.post("/ingest", summary="Ingest transaction or portfolio data")
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
                    "timestamp": tx.timestamp,
                    "features" : tx.features or {},
                })
            except sqlite3.IntegrityError:
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
                    "created_at"     : datetime.utcnow().isoformat(),
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
