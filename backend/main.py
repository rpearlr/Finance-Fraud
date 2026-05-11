from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.config import log, ALLOWED_ORIGINS
from backend.database import init_db, close_db
from backend.state import load_model
from backend.routers import data, ml, regulatory, agent, dashboard, reports, system, frontend, auth

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting FinRisk API...")
    init_db()
    load_model()
    log.info("API ready ✓")
    yield
    close_db()
    log.info("Shutting down FinRisk API")

app = FastAPI(
    title="FinRisk AI Platform",
    description=(
        "Financial Risk & Advisory API — "
        "Fraud Detection · Portfolio Analytics · Regulatory RAG Search · Multi-Agent Query"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include Routers ──────────────────────────────────────────────────────────
app.include_router(data.router)
app.include_router(ml.router)
app.include_router(regulatory.router)
app.include_router(agent.router)
app.include_router(dashboard.router)
app.include_router(reports.router)
app.include_router(system.router)
app.include_router(auth.router)
app.include_router(frontend.router)

app.mount("/static", StaticFiles(directory="frontend"), name="static")