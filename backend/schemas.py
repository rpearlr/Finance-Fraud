from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional

class TransactionIngest(BaseModel):
    tx_id    : str            = Field(..., examples=["TX-94821"])
    amount   : float          = Field(..., gt=0, examples=[2400.50])
    merchant : str            = Field(..., examples=["Online Electronics"])
    timestamp: str            = Field(..., examples=["2025-05-01T14:30:00"])
    features : Optional[dict] = Field(None, description="V1-V28 + engineered features")

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError("timestamp must be ISO 8601: 2025-05-01T14:30:00")
        return v

class PortfolioIngest(BaseModel):
    snapshot_date   : str        = Field(..., examples=["2025-05-01"])
    total_value_inr : float      = Field(..., gt=0, examples=[4820000000.0])
    holdings        : list[dict] = Field(..., examples=[[
        {"asset": "HDFCBANK.NS", "weight": 0.182},
        {"asset": "INFY.NS",     "weight": 0.147},
    ]])

class IngestRequest(BaseModel):
    data_type  : str                         = Field(..., examples=["transaction"])
    transaction: Optional[TransactionIngest] = None
    portfolio  : Optional[PortfolioIngest]   = None

    @field_validator("data_type")
    @classmethod
    def validate_data_type(cls, v: str) -> str:
        if v not in ("transaction", "portfolio"):
            raise ValueError("data_type must be 'transaction' or 'portfolio'")
        return v

class PredictRequest(BaseModel):
    V1 : float; V2 : float; V3 : float; V4 : float; V5 : float
    V6 : float; V7 : float; V8 : float; V9 : float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    amount_zscore : float = Field(..., examples=[1.5])
    velocity_24tx : float = Field(..., examples=[3200.0])
    hour_of_day   : int   = Field(..., ge=0, le=23, examples=[2])
    amount_log    : float = Field(..., examples=[7.8])
    tx_id: Optional[str] = None

class AgentQueryRequest(BaseModel):
    question      : str           = Field(..., max_length=500)
    session_id    : Optional[str] = None
    agent_override: Optional[str] = Field(None, description="Optional agent name to bypass orchestrator routing")

class ReportRequest(BaseModel):
    report_type: str           = Field(..., description="'fraud' or 'investment'")
    query      : Optional[str] = None
    tx_id      : Optional[str] = None
