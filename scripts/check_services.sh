#!/bin/bash
# 서비스 상태 확인 스크립트

echo "📊 Service Status Check"
echo "======================="
echo ""

# SAR Download API
echo "1️⃣ SAR Download API (port 8001):"
if curl -s http://localhost:8001/health > /dev/null 2>&1; then
    echo "   ✅ Running"
else
    echo "   ❌ Not running"
fi

# InSAR Processing API
echo "2️⃣ InSAR Processing API (port 8002):"
if curl -s http://localhost:8002/health > /dev/null 2>&1; then
    echo "   ✅ Running"
else
    echo "   ❌ Not running"
fi

# Agent Server
echo "3️⃣ Agent Server (port 8000):"
if curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo "   ✅ Running"
else
    echo "   ❌ Not running"
fi

echo ""
echo "📋 Process List:"
ps aux | grep -E "sar_download_api|insar_api|server.py" | grep -v grep || echo "  (none)"
