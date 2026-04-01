FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install package and dependencies in one layer.
# Copying pyproject.toml first leverages Docker layer caching.
COPY pyproject.toml requirements.txt ./
RUN pip install --upgrade pip && pip install .

# Copy source last so code changes don't bust the dependency cache.
COPY . .

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
