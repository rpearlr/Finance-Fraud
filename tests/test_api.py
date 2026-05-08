import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data

def test_ingest_transaction():
    payload = {
        "data_type": "transaction",
        "transaction": {
            "tx_id": "TEST-12345",
            "amount": 100.50,
            "merchant": "Test Merchant",
            "timestamp": "2025-05-01T14:30:00",
            "features": {"V1": 0.5, "V2": -0.5}
        }
    }
    response = client.post("/ingest", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["ingested"] == "transaction"

def test_ingest_portfolio():
    payload = {
        "data_type": "portfolio",
        "portfolio": {
            "snapshot_date": "2025-05-01",
            "total_value_inr": 500000.0,
            "holdings": [
                {"asset": "TEST.NS", "weight": 1.0}
            ]
        }
    }
    response = client.post("/ingest", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["ingested"] == "portfolio"

def test_predict_requires_fields():
    payload = {}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Unprocessable Entity due to missing fields
