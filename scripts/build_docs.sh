#!/usr/bin/env bash
set -e

# Build documentation
# - If Docker daemon is running: use `docker compose run backend` to build docs
# - Otherwise: run `sphinx-build` locally (with dependency check)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
DOCS_DIR="${PROJECT_ROOT}/backend/docs"
BACKEND_DIR="${PROJECT_ROOT}/backend"

# Helper: run docs build locally
run_local_build() {
  echo "[build_docs] Docker not available – running docs build locally."
  cd "${BACKEND_DIR}"
  # Ensure dependencies are installed
  if ! python -c "import sphinx_autodoc_typehints" 2>/dev/null; then
    echo "[build_docs] Installing documentation dependencies..."
    pip install -e . > /dev/null 2>&1
  fi
  cd "${DOCS_DIR}"
  echo "[build_docs] Building documentation..."
  sphinx-build -b html . _build/html
}

# Check if Docker daemon is running
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  echo "[build_docs] Docker is running – using Docker container to build docs."
  
  # Detect compose v2 (`docker compose`) or v1 (`docker-compose`)
  if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
  elif command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker-compose"
  else
    echo "[build_docs] Docker Compose not found – falling back to local build."
    run_local_build
    exit 0
  fi
  
  echo "[build_docs] Building documentation in backend container..."
  # Run sphinx-build using the backend service image
  $DOCKER_COMPOSE run --rm backend sh -c "cd docs && sphinx-build -b html . _build/html"
  
else
  # Docker not running → run local build
  run_local_build
fi

echo ""
echo "[build_docs] Docs built successfully!"
echo "Open this in your browser:"
echo "file://${DOCS_DIR}/_build/html/index.html"