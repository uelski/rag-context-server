FROM python:3.11-slim

# Basics
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# System deps (keep minimal; add build-essential only if you compile)
# RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# 1) Install Python deps
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt \
 && pip install --no-cache-dir uvicorn[standard]

# 2) Copy your FastAPI app
# Copy all source (use .dockerignore to keep image small)
COPY . /app

# 3) Make repo root importable (if you ever do `from app...`)
ENV PYTHONPATH=/app

# Note: Cloud Run ignores EXPOSE, binding to 0.0.0.0:${PORT} is what matters
# EXPOSE 8080

# Start FastAPI with Uvicorn
CMD sh -c 'uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}'