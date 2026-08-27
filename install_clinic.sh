#!/bin/bash
# ================================================================
# Sentinel Medical AI — Clinic Installation Script
# ================================================================
# Run this on the clinic's Windows PC (via WSL or Git Bash)
# or Linux machine to set up the complete Sentinel system.
#
# Prerequisites:
#   - Docker Desktop installed
#   - Trained model file (best_model.pt) in models/densenet/
#   - Orthanc configured with clinic's imaging machine IP
#
# This script:
#   1. Configures Orthanc PACS server
#   2. Starts AI inference server
#   3. Starts Orthanc watcher (auto-analysis)
#   4. Sets up local database
#   5. Creates default admin account
#
# Usage:
#   bash install_clinic.sh
# ================================================================

set -e

echo ""
echo "============================================================"
echo "  SENTINEL MEDICAL AI — Clinic Installation"
echo "  SIA Medical AI | siaa.uz"
echo "============================================================"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed!"
    echo "Install Docker Desktop from: https://www.docker.com/products/docker-desktop/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "ERROR: Docker Compose is not installed!"
    exit 1
fi

# Check model file
if [ ! -f "models/densenet/best_model.pt" ]; then
    echo "WARNING: Model file not found at models/densenet/best_model.pt"
    echo "The inference server will start in demo mode."
    echo "Copy your trained model here before production use."
    echo ""
fi

# Generate encryption key if not set
if [ -z "$SENTINEL_DB_KEY" ]; then
    export SENTINEL_DB_KEY=$(openssl rand -hex 32)
    echo "SENTINEL_DB_KEY=$SENTINEL_DB_KEY" > .env
    echo "Generated encryption key (saved to .env)"
fi

# Create required directories
mkdir -p data/{dicom,processed,reports} models/densenet logs

echo ""
echo "[1/4] Starting Orthanc PACS Server..."
echo "  DICOM AET: SENTINEL"
echo "  DICOM Port: 4242 (configure your imaging machine to send here)"
echo "  Web UI: http://localhost:8042 (user: sentinel / pass: sentinel2024)"

echo ""
echo "[2/4] Starting AI Inference Server..."
echo "  API: http://localhost:8000"
echo "  Docs: http://localhost:8000/docs"

echo ""
echo "[3/4] Starting Orthanc Watcher..."
echo "  Auto-analyzes new studies every 10 seconds"

echo ""
echo "[4/4] Building and starting all services..."
docker compose up -d --build

echo ""
echo "============================================================"
echo "  INSTALLATION COMPLETE!"
echo "============================================================"
echo ""
echo "  Services running:"
echo "  ✓ Orthanc PACS:      http://localhost:8042"
echo "  ✓ AI Server:         http://localhost:8000"
echo "  ✓ API Docs:          http://localhost:8000/docs"
echo "  ✓ Watcher:           Running (background)"
echo ""
echo "  Default login:"
echo "    Username: admin"
echo "    Password: sentinel2024"
echo "    ⚠ CHANGE THIS PASSWORD AFTER FIRST LOGIN!"
echo ""
echo "  Configure your imaging machine:"
echo "    DICOM AET: SENTINEL"
echo "    IP: $(hostname -I | awk '{print $1}')"
echo "    Port: 4242"
echo ""
echo "  Now install the Sentinel Desktop App on the"
echo "  radiologist's workstation and connect to this server."
echo ""
echo "============================================================"
