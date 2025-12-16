#!/bin/bash

# Show logs for the InterpolatorApp Docker 
# Usage: ./scripts/docker-logs.sh
set -e

# Detect docker compose (v2 or v1)
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  DOCKER_COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  DOCKER_COMPOSE="docker-compose"
else
  echo "ERROR: Docker Compose is not available. Install Docker Desktop or docker-compose."
  exit 1
fi

echo "[INFO] Showing logs for all services (Ctrl+C to stop)..."
$DOCKER_COMPOSE logs -f