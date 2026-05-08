import random
from datetime import datetime, timedelta
from pymongo import MongoClient, ASCENDING
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME   = os.getenv("MONGO_DB",  "finrisk")

def seed_db():
    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]

    # ── Collections ────────────────────────────────────────────────────────────
    transactions       = db["transactions"]
    portfolio_snapshots = db["portfolio_snapshots"]

    # ── Indexes (idempotent) ───────────────────────────────────────────────────
    transactions.create_index("tx_id", unique=True)
    transactions.create_index([("timestamp", ASCENDING)])
    portfolio_snapshots.create_index([("snapshot_date", ASCENDING)])

    # ── 1. Transactions ────────────────────────────────────────────────────────
    merchants = [
        "Amazon", "Flipkart", "Local Grocery",
        "Uber", "Zomato", "Foreign/Unrecognized", "Electronics Store",
    ]
    now  = datetime.now()
    docs = []

    for _ in range(1200):
        tx_time     = now - timedelta(minutes=random.randint(0, 1440))
        amount      = round(random.uniform(50, 15000), 2)
        merchant    = random.choice(merchants)
        hour        = tx_time.hour
        fraud_score = random.uniform(0.01, 0.3)
        is_fraud    = False

        if merchant == "Foreign/Unrecognized" and amount > 5000:
            is_fraud = random.random() < 0.6
        if hour < 6 and random.random() < 0.2:
            is_fraud = True
        if is_fraud:
            fraud_score = random.uniform(0.51, 0.99)

        docs.append({
            "tx_id"      : f"TX-{random.randint(100000, 999999)}",
            "amount"     : amount,
            "merchant"   : merchant,
            "timestamp"  : tx_time,
            "features"   : {},
            "fraud_score": round(fraud_score, 4),
            "label"      : 1 if fraud_score >= 0.5 else 0,
            "created_at" : tx_time,
        })

    # insert_many with ordered=False skips duplicate tx_ids instead of aborting
    if docs:
        try:
            transactions.insert_many(docs, ordered=False)
        except Exception:
            pass  # bulk write errors from duplicates are fine

    # ── 2. Portfolio snapshot ──────────────────────────────────────────────────
    portfolio_snapshots.delete_many({})   # fresh snapshot each seed run
    assets     = [
        ("HDFCBANK.NS", 0.30),
        ("INFY.NS",     0.20),
        ("RELIANCE.NS", 0.25),
        ("TCS.NS",      0.25),
    ]
    total_val  = 4_820_000_000.0          # 482 Cr INR
    snap_date  = now.strftime("%Y-%m-%d")

    portfolio_snapshots.insert_many([
        {
            "snapshot_date"  : snap_date,
            "total_value_inr": total_val,
            "asset_class"    : asset,
            "weight"         : weight,
            "created_at"     : now,
        }
        for asset, weight in assets
    ])

    count = transactions.count_documents({})
    print(f"✓ Seeded {count} transactions and {len(assets)} portfolio holdings into '{DB_NAME}'")
    client.close()

if __name__ == "__main__":
    seed_db()