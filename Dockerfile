# Raiden API - Fly.io Deployment
# Multi-stage build for smaller image

FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install core dependencies from pyproject.toml (excluding heavy AI extras)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic>=2.0 \
    pydantic-settings>=2.0 \
    typer[all]>=0.9.0 \
    rich>=13.0 \
    pyproj>=3.6 \
    shapely>=2.0 \
    numpy>=1.24 \
    pandas>=2.0 \
    requests>=2.31 \
    httpx>=0.25 \
    pillow>=10.0 \
    opencv-python-headless>=4.8 \
    python-dotenv>=1.0 \
    geomeppy>=0.11 \
    eppy>=0.5 \
    scikit-learn>=1.3.0 \
    scipy>=1.11

# Production image
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/ ./src/
COPY pyproject.toml ./

# Create necessary directories
RUN mkdir -p output/api data/cache

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Run the API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]

