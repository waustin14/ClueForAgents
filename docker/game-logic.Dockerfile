# Wraps the existing game engine + agents behind the FastAPI service in
# service/app.py. Built from repo root: docker build -f docker/game-logic.Dockerfile .
FROM python:3.13-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

# Dependency layer first so code-only edits don't invalidate it.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project \
    --extra api --extra langchain --extra otel --extra langsmith

# Everything the service actually imports at runtime — no tests/, no docker/.
COPY telemetry.py main.py ./
COPY models/ ./models/
COPY game/ ./game/
COPY agents/ ./agents/
COPY transport/ ./transport/
COPY service/ ./service/

EXPOSE 8000

# 127.0.0.1, not localhost: uvicorn binds 0.0.0.0 (IPv4-only) below, and
# "localhost" resolving to ::1 first would make this probe IPv6-first too
# (see docker/frontend.Dockerfile for where that actually bit the frontend).
HEALTHCHECK --interval=5s --timeout=3s --start-period=5s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)" || exit 1

CMD ["uv", "run", "--no-sync", "uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
