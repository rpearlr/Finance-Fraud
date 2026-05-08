import pytest
from agents.fraud_agent import get_risk_level
from agents.rag_pipeline import infer_source_type, chunk_text

def test_fraud_risk_levels():
    assert get_risk_level(0.95)[0] == "HIGH"
    assert get_risk_level(0.65)[0] == "MEDIUM"
    assert get_risk_level(0.15)[0] == "LOW"
    assert get_risk_level(0.0)[0] == "LOW"
    assert get_risk_level(1.0)[0] == "HIGH"

def test_rag_source_inference():
    assert infer_source_type("sebi_regulation_2024.pdf") == "SEBI"
    assert infer_source_type("rbi_circular.pdf") == "RBI"
    assert infer_source_type("basel_iii_guidelines.pdf") == "Basel"
    assert infer_source_type("random_doc.pdf") == "General"

def test_rag_chunking():
    text = "This is a long piece of text that needs to be chunked. " * 20
    source = "test_doc"
    source_type = "General"
    chunks = chunk_text(text, source, source_type)
    
    assert len(chunks) > 0
    assert chunks[0]["source"] == source
    assert chunks[0]["source_type"] == source_type
    assert "content" in chunks[0]

def test_investment_agent_import():
    from agents.investment_agent import InvestmentAdviceAgent
    agent = InvestmentAdviceAgent()
    assert agent is not None
