#!/bin/bash

# Docker stop script for Interpolator App
# Usage: ./scripts/stop_app.sh

set -e

# Colors for output
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
BLUE="\033[0;34m"
NC="\033[0m" # No Color

# Function to print colored output
print_status() {
  echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
  echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

# Check if docker compose is available (try v2 first, then v1)
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  DOCKER_COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  DOCKER_COMPOSE="docker-compose"
else
  print_error "Docker Compose is not available. Please install Docker Desktop or docker-compose and try again."
  exit 1
fi

print_status "Stopping Interpolator App containers..."

# Stop and remove containers
$DOCKER_COMPOSE down --remove-orphans

print_success "Interpolator App containers stopped successfully!"
