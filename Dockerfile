# Production-ready Dockerfile optimized for 2 vCPU / 8GB RAM
# Multi-stage build: builder compiles wheels, runtime installs them cleanly

# ============================================================================
# Stage 1: Builder - compile Python wheels
# ============================================================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy everything needed to build the package
COPY pyproject.toml ./
COPY src/ ./src/

# Build wheels for the package and all its dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels .

# ============================================================================
# Stage 2: Runtime - lean production image
# ============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Non-root user for security
RUN groupadd -r uvicorn && useradd -r -g uvicorn uvicorn

# Copy pre-built wheels from builder stage
COPY --from=builder /build/wheels /tmp/wheels

# Install all dependencies from local wheels only — no src/ needed here
# We use --no-index so pip only looks at /tmp/wheels and never hits PyPI
RUN pip install --upgrade pip && \
    pip install --no-index --find-links=/tmp/wheels /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels

# Now copy application code (after deps, so code changes don't bust dep layer)
COPY src/ ./src/
COPY server/ ./server/
COPY inference.py ./

# Make src importable as installed package
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

USER uvicorn

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health', timeout=3)" || exit 1

CMD ["uvicorn", \
    "server.app:app", \
    "--host", "0.0.0.0", \
    "--port", "7860", \
    "--workers", "2", \
    "--loop", "uvloop", \
    "--http", "httptools", \
    "--timeout-keep-alive", "60", \
    "--no-access-log"]