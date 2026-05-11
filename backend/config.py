import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("finrisk")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
MODEL_PATH    = BASE_DIR / "ml" / "models" / "fraud_model.pkl"
FEATURES_PATH = BASE_DIR / "ml" / "models" / "feature_names.txt"
FORECAST_PATH = BASE_DIR / "data" / "staged" / "portfolio_forecast.csv"

# ── SQLite ─────────────────────────────────────────────────────────────────────
SQLITE_PATH = os.getenv("SQLITE_PATH", str(BASE_DIR / "finrisk.db"))
ALLOWED_ORIGINS = "*"

# ── Auth (JWT) ────────────────────────────────────────────────────────────────
JWT_SECRET = os.getenv("JWT_SECRET", "980962873db841f32a82208b088e5d023b6b12f6d2f3c7e0980962873db841f3")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # 24 hours
