#!/usr/bin/env bash

# Run backend test suite.
# - If Docker is running: use 'docker compose run backend pytest -sv tests'
# - Otherwise: run 'pytest -sv tests' locally from backend/

set -euo pipefail

# Find repo root (script is in ./scripts/)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[run_tests] Project root: $ROOT_DIR"

# Helper: run tests locally
run_local_tests() {
  echo "[run_tests] Docker not available or compose missing - running tests locally."
  cd "$ROOT_DIR/backend"
  pytest -sv tests
}

# Check if Docker is running
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  echo "[run_tests] Docker is running - trying to use docker compose."

  # Detect compose v2 (`docker compose`) or v1 (`docker-compose`)
  if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
  elif command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker-compose"
  else
    echo "[run_tests] Docker Compose not found - falling back to local pytest."
    run_local_tests
    exit 0
  fi

  echo "[run_tests] Running tests in backend container..."
  # Run pytest using the backend service image
  $DOCKER_COMPOSE run --rm backend pytest -sv tests

else
  # Docker not running → just run local pytest
  run_local_tests
fi
