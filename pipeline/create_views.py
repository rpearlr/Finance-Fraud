import sqlite3
from pathlib import Path

conn = sqlite3.connect("data/finrisk.db")

# View 1 — fraud summary by hour (feeds the time-series chart)
conn.execute("""
    CREATE VIEW IF NOT EXISTS v_fraud_hourly AS
    SELECT
        strftime('%Y-%m-%d %H:00', timestamp)  AS hour_bucket,
        COUNT(*)                                AS total_tx,
        SUM(CASE WHEN label=1 THEN 1 ELSE 0 END) AS fraud_count,
        AVG(fraud_score)                        AS avg_fraud_score,
        MAX(fraud_score)                        AS max_fraud_score
    FROM transactions
    WHERE fraud_score IS NOT NULL
    GROUP BY hour_bucket
""")

# View 2 — risk tier breakdown (feeds the donut chart)
conn.execute("""
    CREATE VIEW IF NOT EXISTS v_risk_tiers AS
    SELECT
        CASE
            WHEN fraud_score >= 0.80 THEN 'HIGH'
            WHEN fraud_score >= 0.50 THEN 'MEDIUM'
            ELSE 'LOW'
        END AS risk_tier,
        COUNT(*) AS count,
        AVG(amount) AS avg_amount
    FROM transactions
    WHERE fraud_score IS NOT NULL
    GROUP BY risk_tier
""")

# View 3 — agent query log (feeds Agent Insights tab)
conn.execute("""
    CREATE VIEW IF NOT EXISTS v_agent_insights AS
    SELECT
        created_at,
        agent_used,
        question,
        substr(answer, 1, 200) AS answer_preview,
        latency_ms
    FROM agent_queries
    ORDER BY created_at DESC
    LIMIT 100
""")

conn.commit()
conn.close()
print("Views created")