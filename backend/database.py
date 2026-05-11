import sqlite3
import json
from datetime import datetime
from typing import Optional, Any, List, Dict
from backend.config import SQLITE_PATH, log

_db_conn: sqlite3.Connection = None

def get_conn() -> sqlite3.Connection:
    global _db_conn
    if _db_conn is None:
        init_db()
    return _db_conn

def query_db(query: str, params: tuple = ()) -> List[Dict[str, Any]]:
    """Execute a SELECT query and return results as a list of dictionaries."""
    conn = get_conn()
    try:
        cur = conn.execute(query, params)
        rows = [dict(r) for r in cur.fetchall()]
        
        # Automatic JSON deserialization for columns that look like JSON
        for r in rows:
            for k, v in r.items():
                if isinstance(v, str) and (v.startswith('{') or v.startswith('[')):
                    try:
                        r[k] = json.loads(v)
                    except: pass
        return rows
    except Exception as e:
        log.error(f"SQL Query Error: {e}\nQuery: {query}\nParams: {params}")
        raise

def query_one(query: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
    """Execute a SELECT query and return the first result as a dictionary."""
    rows = query_db(query, params)
    return rows[0] if rows else None

def execute_db(query: str, params: tuple = ()) -> int:
    """Execute an INSERT/UPDATE/DELETE query and return the last inserted row ID."""
    conn = get_conn()
    try:
        # Handle list/dict params by serializing to JSON
        processed_params = []
        for p in params:
            if isinstance(p, (dict, list)):
                processed_params.append(json.dumps(p))
            else:
                processed_params.append(p)
                
        cur = conn.execute(query, tuple(processed_params))
        conn.commit()
        return cur.lastrowid
    except Exception as e:
        log.error(f"SQL Execution Error: {e}\nQuery: {query}\nParams: {params}")
        raise

def execute_many_db(query: str, params_list: List[tuple]) -> None:
    """Execute many INSERT/UPDATE/DELETE queries."""
    conn = get_conn()
    try:
        processed_list = []
        for params in params_list:
            processed_params = []
            for p in params:
                if isinstance(p, (dict, list)):
                    processed_params.append(json.dumps(p))
                else:
                    processed_params.append(p)
            processed_list.append(tuple(processed_params))
            
        conn.executemany(query, processed_list)
        conn.commit()
    except Exception as e:
        log.error(f"SQL ExecuteMany Error: {e}\nQuery: {query}")
        raise

def init_db() -> None:
    global _db_conn
    _db_conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    _db_conn.row_factory = sqlite3.Row
    
    _db_conn.executescript("""
        CREATE TABLE IF NOT EXISTS transactions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            tx_id        TEXT    UNIQUE NOT NULL,
            timestamp    TEXT    NOT NULL,
            amount       REAL    NOT NULL,
            merchant     TEXT,
            category     TEXT,
            fraud_score  REAL,
            label        INTEGER,
            features     TEXT,   -- JSON string
            inserted     TEXT    DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_tx_timestamp ON transactions(timestamp);
        CREATE INDEX IF NOT EXISTS idx_tx_label     ON transactions(label);

        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date  TEXT    NOT NULL,
            total_value_inr REAL,
            asset_class    TEXT,
            weight         REAL,
            holdings       TEXT,   -- JSON string
            inserted       TEXT    DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_pf_date ON portfolio_snapshots(snapshot_date);

        CREATE TABLE IF NOT EXISTS agent_queries (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            question     TEXT,
            agent_used   TEXT,
            answer       TEXT,
            sources      TEXT,   -- JSON string
            chunks_used  INTEGER,
            tokens_used  INTEGER,
            latency_ms   INTEGER,
            created_at   TEXT    DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_aq_created ON agent_queries(created_at);

        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT    UNIQUE NOT NULL,
            email      TEXT    UNIQUE NOT NULL,
            password   TEXT    NOT NULL,
            salt       TEXT    NOT NULL,
            created_at TEXT    DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
    """)
    _db_conn.commit()
    log.info(f"Relational SQLite initialized at {SQLITE_PATH}")

def close_db() -> None:
    global _db_conn
    if _db_conn:
        _db_conn.close()
        _db_conn = None

