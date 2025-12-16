#!/usr/bin/env bash

# Run benchmark and plot results.
# - If Docker is running: use 'docker compose run backend' to run the scripts
# - Otherwise: run locally from backend/

set -euo pipefail

# Find repo root (script is in ./scripts/)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[run_benchmark] Project root: $ROOT_DIR"

# Helper: run benchmarks locally
run_local_benchmarks() {
  echo "[run_benchmark] Docker not available or compose missing - running benchmarks locally."
  cd "$ROOT_DIR/backend"

  echo "[run_benchmark] Running benchmark.py..."
  python analysis/benchmark.py

  echo "[run_benchmark] Running plot_results.py..."
  python analysis/plot_results.py

  echo "[run_benchmark] Benchmarks complete!"
}

# Check if Docker daemon is running
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  echo "[run_benchmark] Docker is running - trying to use docker compose."

  # Detect compose v2 (`docker compose`) or v1 (`docker-compose`)
  if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
  elif command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker-compose"
  else
    echo "[run_benchmark] Docker Compose not found - falling back to local execution."
    run_local_benchmarks
    exit 0
  fi

  echo "[run_benchmark] Running benchmark.py in backend container..."
  $DOCKER_COMPOSE run --rm backend python analysis/benchmark.py

  echo "[run_benchmark] Running plot_results.py in backend container..."
  $DOCKER_COMPOSE run --rm backend python analysis/plot_results.py

  echo "[run_benchmark] Benchmarks complete!"

else
  # Docker not running → just run locally
  run_local_benchmarks
fi
