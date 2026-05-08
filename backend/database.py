import sqlite3
import json
import logging
from datetime import datetime
from typing import Optional
from backend.config import SQLITE_PATH, log

_db_conn: sqlite3.Connection = None

def get_conn() -> sqlite3.Connection:
    global _db_conn
    return _db_conn

def col(name: str):
    return _Table(name, get_conn())

class _Table:
    def __init__(self, name: str, conn: sqlite3.Connection):
        self._name = name
        self._conn = conn

    def _ensure(self):
        """Create the backing table if it doesn't exist yet."""
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._name} (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                data      TEXT    NOT NULL,
                inserted  TEXT    DEFAULT (datetime('now'))
            )
        """)
        self._conn.commit()

    def _rows(self, where: str = "1=1", params=()) -> list[dict]:
        self._ensure()
        cur = self._conn.execute(
            f"SELECT data FROM {self._name} WHERE {where}", params
        )
        return [json.loads(r[0]) for r in cur.fetchall()]

    def _insert(self, doc: dict) -> None:
        self._ensure()
        self._conn.execute(
            f"INSERT INTO {self._name} (data) VALUES (?)",
            (json.dumps(doc, default=str),),
        )
        self._conn.commit()

    def create_index(self, *args, **kwargs):
        pass

    def insert_one(self, doc: dict):
        if self._name == "transactions" and "tx_id" in doc:
            existing = self.find_one({"tx_id": doc["tx_id"]})
            if existing:
                raise sqlite3.IntegrityError(f"UNIQUE constraint failed: tx_id={doc['tx_id']}")
        self._insert(doc)

    def insert_many(self, docs: list[dict]):
        self._ensure()
        self._conn.executemany(
            f"INSERT INTO {self._name} (data) VALUES (?)",
            [(json.dumps(d, default=str),) for d in docs],
        )
        self._conn.commit()

    def find_one(self, filter_: dict = None, sort: list = None) -> Optional[dict]:
        rows = self._rows()
        if filter_:
            rows = [r for r in rows if self._match(r, filter_)]
        if sort:
            rows = self._sort(rows, sort)
        return rows[0] if rows else None

    def find(self, filter_: dict = None, projection: dict = None,
             sort: list = None, limit: int = 0) -> list[dict]:
        rows = self._rows()
        if filter_:
            rows = [r for r in rows if self._match(r, filter_)]
        if sort:
            rows = self._sort(rows, sort)
        if limit:
            rows = rows[:limit]
        if projection:
            rows = [self._project(r, projection) for r in rows]
        return rows

    def count_documents(self, filter_: dict = None) -> int:
        rows = self._rows()
        if filter_:
            rows = [r for r in rows if self._match(r, filter_)]
        return len(rows)

    def update_one(self, filter_: dict, update: dict):
        self._ensure()
        rows_raw = self._conn.execute(
            f"SELECT id, data FROM {self._name}"
        ).fetchall()
        for row_id, raw in rows_raw:
            doc = json.loads(raw)
            if self._match(doc, filter_):
                if "$set" in update:
                    doc.update(update["$set"])
                self._conn.execute(
                    f"UPDATE {self._name} SET data=? WHERE id=?",
                    (json.dumps(doc, default=str), row_id),
                )
                self._conn.commit()
                return

    def distinct(self, field: str, filter_: dict = None) -> list:
        rows = self._rows()
        if filter_:
            rows = [r for r in rows if self._match(r, filter_)]
        return list({r.get(field) for r in rows if field in r})

    def aggregate(self, pipeline: list) -> list:
        rows = self._rows()
        for stage in pipeline:
            if "$match" in stage:
                rows = [r for r in rows if self._match(r, stage["$match"])]
            elif "$group" in stage:
                spec   = stage["$group"]
                id_key = spec.get("_id")
                groups: dict = {}
                for r in rows:
                    if id_key is None:
                        gkey = None
                    elif isinstance(id_key, str) and id_key.startswith("$"):
                        gkey = r.get(id_key[1:])
                    elif isinstance(id_key, dict):
                        op, field = next(iter(id_key.items()))
                        val = r.get(field[1:]) if isinstance(field, str) and field.startswith("$") else field
                        if op == "$hour":
                            try:
                                dt = datetime.fromisoformat(str(val)) if isinstance(val, str) else val
                                gkey = dt.hour if isinstance(dt, datetime) else None
                            except Exception:
                                gkey = None
                        else:
                            gkey = val
                    else:
                        gkey = id_key
                    if gkey not in groups:
                        groups[gkey] = {"_id": gkey, "_vals": {}}
                    for acc_field, acc_expr in spec.items():
                        if acc_field == "_id":
                            continue
                        if isinstance(acc_expr, dict):
                            agg_op, src = next(iter(acc_expr.items()))
                            src_field   = src[1:] if isinstance(src, str) and src.startswith("$") else src
                            src_val     = r.get(src_field) if isinstance(src_field, str) else None
                            if src_val is None:
                                continue
                            bucket = groups[gkey]["_vals"].setdefault(acc_field, [])
                            bucket.append(src_val)
                result = []
                for gkey, grp in groups.items():
                    doc = {"_id": gkey}
                    for acc_field, vals in grp["_vals"].items():
                        acc_expr = spec.get(acc_field, {})
                        agg_op   = next(iter(acc_expr)) if isinstance(acc_expr, dict) else None
                        if agg_op == "$avg":
                            doc[acc_field] = sum(vals) / len(vals) if vals else 0
                        elif agg_op == "$max":
                            doc[acc_field] = max(vals) if vals else 0
                        elif agg_op == "$sum":
                            doc[acc_field] = sum(vals)
                        else:
                            doc[acc_field] = vals[-1] if vals else None
                    result.append(doc)
                rows = result
            elif "$sort" in stage:
                rows = self._sort(rows, list(stage["$sort"].items()))
            elif "$limit" in stage:
                rows = rows[: stage["$limit"]]
        return rows

    @staticmethod
    def _match(doc: dict, filter_: dict) -> bool:
        for key, val in filter_.items():
            if isinstance(val, dict):
                doc_val = doc.get(key)
                for op, operand in val.items():
                    if op == "$exists":
                        if operand and doc_val is None: return False
                        if not operand and doc_val is not None: return False
                    elif op == "$gt":
                        if doc_val is None or doc_val <= operand: return False
                    elif op == "$gte":
                        if doc_val is None or doc_val < operand: return False
                    elif op == "$lt":
                        if doc_val is None or doc_val >= operand: return False
                    elif op == "$lte":
                        if doc_val is None or doc_val > operand: return False
                    elif op == "$ne":
                        if doc_val == operand: return False
                    elif op == "$in":
                        if doc_val not in operand: return False
            else:
                if doc.get(key) != val: return False
        return True

    @staticmethod
    def _sort(rows: list, sort_spec: list) -> list:
        for field, direction in reversed(sort_spec):
            rows = sorted(
                rows,
                key=lambda r: (r.get(field) is None, r.get(field)),
                reverse=(direction == -1),
            )
        return rows

    @staticmethod
    def _project(doc: dict, projection: dict) -> dict:
        include = {k for k, v in projection.items() if v}
        exclude = {k for k, v in projection.items() if not v}
        if include:
            return {k: v for k, v in doc.items() if k in include}
        return {k: v for k, v in doc.items() if k not in exclude}

def init_db() -> None:
    global _db_conn
    _db_conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    _db_conn.row_factory = sqlite3.Row
    _db_conn.executescript("""
        CREATE TABLE IF NOT EXISTS transactions (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            data      TEXT    NOT NULL,
            inserted  TEXT    DEFAULT (datetime('now'))
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_tx_id
            ON transactions(json_extract(data, '$.tx_id'));
        CREATE INDEX IF NOT EXISTS idx_tx_timestamp
            ON transactions(json_extract(data, '$.timestamp'));
        CREATE INDEX IF NOT EXISTS idx_tx_label
            ON transactions(json_extract(data, '$.label'));
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            data      TEXT    NOT NULL,
            inserted  TEXT    DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_pf_date
            ON portfolio_snapshots(json_extract(data, '$.snapshot_date'));
        CREATE TABLE IF NOT EXISTS agent_queries (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            data      TEXT    NOT NULL,
            inserted  TEXT    DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_aq_created
            ON agent_queries(json_extract(data, '$.created_at'));
    """)
    _db_conn.commit()
    log.info(f"SQLite ready — {SQLITE_PATH}")

def close_db() -> None:
    global _db_conn
    if _db_conn:
        _db_conn.close()
        _db_conn = None
