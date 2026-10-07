# Production Dockerfile for Visual Defect Detection API Service
FROM python:3.11-slim

# Set environment variables for Python performance and safety
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies (curl for healthcheck, libgl for image processing if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY requirements.txt .

# Install locked dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, API modules, and trained model checkpoints
COPY src/ ./src/
COPY api/ ./api/
COPY checkpoints/ ./checkpoints/

# Expose standard inference service port
EXPOSE 8000

# Docker healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start Uvicorn ASGI server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
