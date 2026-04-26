# VerdictEnv — production Docker image
# Gradio UI + FastAPI REST API served together via uvicorn.
# Gradio UI  → http://localhost:7860/
# API docs   → http://localhost:7860/docs
# API routes → http://localhost:7860/api/*

FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

# ── Layer 1: install dependencies (cached unless requirements.txt changes) ──
COPY server/requirements.txt server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

# ── Layer 2: copy source and install the package (no-deps, already above) ──
COPY . .
RUN pip install --no-cache-dir --no-deps .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/api/health')"

CMD ["python", "-m", "verdict_env.server.app"]
