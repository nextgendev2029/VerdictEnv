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
    PORT=7860 \
    PYTHONPATH=/app

# ── Layer 1: install dependencies (cached unless requirements.txt changes) ──
COPY server/requirements.txt server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

# ── Layer 2: copy source ──
COPY . .

# ── Layer 3: create a proper verdict_env/ package Python can actually find ──
# The source layout has package files directly at the repo root (not inside a
# verdict_env/ subdirectory), which confuses pip's wheel/editable installers.
# The fix: build the expected directory layout explicitly, no pip install needed.
RUN mkdir -p verdict_env/server \
    && cp __init__.py models.py tasks.py inference.py client.py verdict_env/ \
    && cp -r server/. verdict_env/server/

# With PYTHONPATH=/app, Python resolves:
#   verdict_env            → /app/verdict_env/
#   verdict_env.server.app → /app/verdict_env/server/app.py  ✓

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/api/health')"

CMD ["python", "-m", "verdict_env.server.app"]
