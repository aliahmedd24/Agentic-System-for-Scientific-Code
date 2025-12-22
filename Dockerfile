# Scientific Paper Analysis System
# Multi-stage Docker build for production deployment

# ============================================================================
# Stage 1: Python dependencies
# ============================================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt


# ============================================================================
# Stage 2: Runtime image
# ============================================================================
FROM python:3.11-slim as runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Copy wheels from builder and install
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/outputs /app/reports/generated /app/sandbox \
    && chown -R appuser:appuser /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default command: run server
CMD ["python", "main.py", "server", "--host", "0.0.0.0", "--port", "8000"]


# ============================================================================
# Stage 3: Development image (optional)
# ============================================================================
FROM runtime as development

USER root

# Install development dependencies
RUN pip install pytest pytest-asyncio pytest-cov black isort mypy

USER appuser

# Development command with reload
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
