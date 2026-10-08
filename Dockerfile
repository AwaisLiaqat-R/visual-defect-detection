# ─── Production Dockerfile — Visual Defect Detection API ───────────────────────
# Uses CPU-only PyTorch wheels to keep image size under 1 GB for Render free tier
FROM python:3.11-slim

# Performance & safety env vars
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install CPU-only PyTorch first (avoids downloading CUDA wheels)
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/
COPY api/ ./api/
COPY checkpoints/ ./checkpoints/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start server — PORT env var is respected by Render
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
