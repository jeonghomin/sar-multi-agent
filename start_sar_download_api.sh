#!/bin/bash
# SAR Download API 서버 시작 (agent_cv 루트에서 실행)

echo "🚀 Starting SAR Download API Server..."
echo ""

cd "$(dirname "$0")/sar_api"
./start_sar_api.sh
