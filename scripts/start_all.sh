#!/bin/bash
# 모든 서비스 시작 스크립트

PROJECT_ROOT="/home/mjh/sar-multi-agent"

echo "🚀 Starting All Services..."
echo ""

# 1. SAR Download Service (port 8001)
echo "1️⃣ Starting SAR Download Service (port 8001)..."
cd "$PROJECT_ROOT/services/sar_download"
bash start_sar_api.sh &
SAR_DOWNLOAD_PID=$!
sleep 2

# 2. InSAR Processing Service (port 8002)
echo "2️⃣ Starting InSAR Processing Service (port 8002)..."
cd "$PROJECT_ROOT/services/insar_processing"
bash start_insar_api.sh &
INSAR_PID=$!
sleep 2

# 3. Agent Server (LangGraph + LangServe)
echo "3️⃣ Starting Agent Server (port 8000)..."
cd "$PROJECT_ROOT"
python server.py &
AGENT_PID=$!
sleep 2

echo ""
echo "✅ All services started!"
echo ""
echo "📊 Service Status:"
echo "  - SAR Download API: http://localhost:8001 (PID: $SAR_DOWNLOAD_PID)"
echo "  - InSAR Processing API: http://localhost:8002 (PID: $INSAR_PID)"
echo "  - Agent Server: http://localhost:8000 (PID: $AGENT_PID)"
echo ""
echo "💡 To stop all services: bash scripts/stop_all.sh"
echo ""
echo "Press Ctrl+C to view logs..."

# Wait for all processes
wait
