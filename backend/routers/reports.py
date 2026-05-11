from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from backend.schemas import ReportRequest
from backend.config import log
from backend.state import get_state
from backend.pdf_generator import build_fraud_report, build_investment_report
from backend.utils import _extract_tx_context

router = APIRouter(tags=["Reports"])

@router.post("/report/generate")
def generate_report(req: ReportRequest):
    state = get_state()
    fraud_agent = state["fraud_agent"]
    investment_agent = state["investment_agent"]

    try:
        if req.report_type == "fraud":
            if not req.tx_id:
                raise HTTPException(status_code=400, detail="tx_id required for fraud report")
            
            # Dynamically extract context instead of hardcoded values
            tx_id, fraud_score, features = _extract_tx_context(req.tx_id)
            
            result = fraud_agent.explain(
                fraud_score=fraud_score,
                features=features,
                tx_id=tx_id,
            )
            pdf_buffer = build_fraud_report(tx_id, result.get("fraud_score", fraud_score), result.get("explanation", ""))
            filename   = f"Fraud_Report_{tx_id}.pdf"

        elif req.report_type == "investment":
            if not req.query:
                raise HTTPException(status_code=400, detail="query required for investment report")
            result     = investment_agent.advise(req.query)
            pdf_buffer = build_investment_report(req.query, result.get("explanation", ""))
            filename   = "Investment_Advice_Report.pdf"

        else:
            raise HTTPException(status_code=400, detail="Invalid report_type")

        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))
