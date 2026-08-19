# Multi-stage Dockerfile for FastAPI backend
FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Development stage
FROM base AS development
# Copy dependency file first
COPY pyproject.toml ./

# Install PyTorch CPU-only first (much smaller file size and significantly faster) as we are only using CPU
# BuildKit cache mount preserves downloads between builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy minimal package structure
COPY fivedreg/ ./fivedreg/
COPY main.py ./

# Install remaining dependencies (torch already installed, will be skipped)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e .

# Copy rest of project files
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base AS production
# Copy dependency file first
COPY pyproject.toml ./

# Install PyTorch CPU-only first (much smaller file size and significantly faster) as we are only using CPU
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy minimal package structure
COPY fivedreg/ ./fivedreg/
COPY main.py ./

# Install remaining dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install .

# Copy rest of project files
COPY . .

# Create non-root user
RUN addgroup --system --gid 1001 appgroup && \
    adduser --system --uid 1001 --gid 1001 --no-create-home appuser

# Change ownership of the app directory
RUN chown -R appuser:appgroup /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]