# pipeline/export_for_powerbi.py — run this to refresh Power BI data
import sqlite3
import pandas as pd

conn = sqlite3.connect("data/finrisk.db")
pd.read_sql("SELECT * FROM transactions", conn).to_csv(r"C:\Users\User\Desktop\fin-prj\data\powerbi\transactions.csv", index=False)
pd.read_sql("SELECT * FROM v_fraud_hourly", conn).to_csv(r"C:\Users\User\Desktop\fin-prj\data\powerbi\fraud_hourly.csv", index=False)
pd.read_sql("SELECT * FROM v_risk_tiers", conn).to_csv(r"C:\Users\User\Desktop\fin-prj\data\powerbi\risk_tiers.csv", index=False)
pd.read_sql("SELECT * FROM v_agent_insights", conn).to_csv(r"C:\Users\User\Desktop\fin-prj\data\powerbi\agent_insights.csv", index=False)
pd.read_csv("data/staged/portfolio_forecast.csv").to_csv(r"C:\Users\User\Desktop\fin-prj\data\powerbi\portfolio_forecast.csv", index=False)
conn.close()
print("Exported for Power BI")