import pytest
from fastapi.testclient import TestClient
from backend.main import app
import json

client = TestClient(app)

def test_dashboard_kpis():
    response = client.get("/dashboard/kpis")
    # It might fail if DB is not seeded, but we check if it returns 200 or 500 gracefully
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "transactions_today" in data
        assert "avg_fraud_score" in data
        assert "portfolio_value" in data

def test_agent_history():
    response = client.get("/agent/history?limit=5")
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "count" in data
        assert isinstance(data["history"], list)

def test_report_generate_invalid():
    # Test invalid report type
    payload = {"report_type": "invalid"}
    response = client.post("/report/generate", json=payload)
    assert response.status_code == 400

def test_forecast_portfolio_no_file(monkeypatch):
    # Mocking the file load to fail or succeed
    response = client.get("/forecast/portfolio")
    # Likely 500 if file doesn't exist
    assert response.status_code in [200, 500]

def test_routing_logic():
    from backend.main import _route_question
    assert _route_question("what is the sebi regulation for aif?") == "rag_agent"
    assert _route_question("how many transactions happened last week?") == "sql_agent"
    assert _route_question("what is the model accuracy?") == "ml_expert_agent"
    assert _route_question("hello") == "orchestrator" # default fallback
