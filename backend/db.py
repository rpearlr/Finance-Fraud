"""
backend/db.py  —  MongoDB helpers (drop-in replacement for sqlite3 calls in main.py)

Usage:
    from backend.db import get_db, col

    db  = get_db()                          # pymongo Database
    txs = col("transactions")               # pymongo Collection shortcut
"""

import os
from functools import lru_cache
from pymongo import MongoClient, ASCENDING

MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017")   # 'mongo' = docker service name
DB_NAME   = os.getenv("MONGO_DB",  "finrisk")


@lru_cache(maxsize=1)
def _client() -> MongoClient:
    return MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)


def get_db():
    return _client()[DB_NAME]


def col(name: str):
    return get_db()[name]


def ensure_indexes():
    """Call once at startup."""
    col("transactions").create_index("tx_id", unique=True)
    col("transactions").create_index([("timestamp", ASCENDING)])
    col("transactions").create_index([("label", ASCENDING)])
    col("portfolio_snapshots").create_index([("snapshot_date", ASCENDING)])
    col("agent_queries").create_index([("created_at", ASCENDING)])