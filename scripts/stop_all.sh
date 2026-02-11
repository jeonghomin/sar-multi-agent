#!/bin/bash
# 모든 서비스 종료 스크립트

echo "🛑 Stopping All Services..."

# SAR Download API (port 8001)
echo "1️⃣ Stopping SAR Download Service..."
pkill -f "sar_download_api.py"

# InSAR Processing API (port 8002)
echo "2️⃣ Stopping InSAR Processing Service..."
pkill -f "insar_api.py"

# Agent Server (port 8000)
echo "3️⃣ Stopping Agent Server..."
pkill -f "server.py"

sleep 1

echo ""
echo "✅ All services stopped!"
echo ""
echo "📊 Remaining processes:"
ps aux | grep -E "sar_download_api|insar_api|server.py" | grep -v grep || echo "  (none)"
