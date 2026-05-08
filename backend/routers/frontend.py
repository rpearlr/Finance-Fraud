from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(tags=["Frontend"])

@router.get("/")
async def serve_login():
    return FileResponse("frontend/finrisk_login.html")

@router.get("/dashboard")
async def serve_dashboard():
    return FileResponse("frontend/finrisk_kpi_dashboard.html")

@router.get("/{filename}.html")
async def serve_html_files(filename: str):
    file_path = Path("frontend") / f"{filename}.html"
    if file_path.exists():
        return FileResponse(file_path)
    raise HTTPException(status_code=404)
