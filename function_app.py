import azure.functions as func
from backend.main import app   # your existing FastAPI app

# Azure Functions wraps your existing FastAPI app
# No changes needed to backend/main.py
from azure.functions import AsgiFunctionApp

app_func = AsgiFunctionApp(
    app=app,
    http_auth_level=func.AuthLevel.ANONYMOUS
)