FROM python:3.11-slim

# Layer 1 — system deps (almost never changes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ unixodbc-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Layer 2 — HEAVY ML packages (slowest, install separately so they cache independently)
# sentence-transformers pulls full PyTorch — isolate it so a requirements.txt
# change doesn't force a re-download of 2GB of wheels
COPY requirements-heavy.txt .
RUN pip install --no-cache-dir -r requirements-heavy.txt

# Layer 3 — everything else (fast)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Layer 4 — static data (changes less often than code)
COPY ml/models/      ./ml/models/
COPY data/faiss_index/ ./data/faiss_index/
COPY data/staged/      ./data/staged/
COPY frontend/ ./frontend/

# Layer 5 — app code (changes every push — keep last so above layers stay cached)
COPY backend/   ./backend/
COPY agents/    ./agents/
COPY seed_db.py /app/seed_db.py 
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]