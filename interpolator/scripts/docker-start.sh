#!/usr/bin/env bash
# scripts/run_app.sh
# Build and start backend + frontend with Docker Compose

set -e

# coloured output
BLUE="\033[0;34m"
GREEN="\033[0;32m"
RED="\033[0;31m"
NC="\033[0m" # No Color

info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# --- Go to project root (one level up from this script) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
cd "$PROJECT_ROOT"

# --- Check Docker is installed and running ---
if ! command -v docker >/dev/null 2>&1; then
  error "Docker is not installed or not in PATH."
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  error "Docker is not running. Start Docker Desktop and try again."
  exit 1
fi

# --- Detect docker compose command (v2 or v1) ---
if docker compose version >/dev/null 2>&1; then
  DOCKER_COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  DOCKER_COMPOSE="docker-compose"
else
  error "Docker Compose is not available."
  exit 1
fi

info "Building and starting the stack (backend + frontend)..."

# Enable Docker BuildKit for better caching and faster builds
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

$DOCKER_COMPOSE up --build

# When this exits, containers have stopped
success "Stack stopped."