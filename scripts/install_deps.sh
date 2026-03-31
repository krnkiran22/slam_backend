#!/usr/bin/env bash
set -euo pipefail

echo "=== Build AI Pipeline — Dependency Installer ==="

# System packages
echo "[1/5] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-dev python3-venv python3-pip \
    build-essential cmake git \
    libopencv-dev libeigen3-dev \
    docker.io docker-compose-plugin \
    nodejs npm

# Python venv
echo "[2/5] Setting up Python environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Docker services
echo "[3/5] Starting local services..."
docker compose up -d

# Dashboard
echo "[4/5] Setting up dashboard..."
cd ../frontend
npm install
cd ../backend

# .env
echo "[5/5] Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example — edit it with your values"
fi

echo ""
echo "=== Setup complete ==="
echo "  Start API:       cd backend && uvicorn api.main:app --reload --port 8000"
echo "  Start worker:    cd backend && celery -A api.tasks worker --loglevel=info"
echo "  Start dashboard: cd frontend && npm run dev"
