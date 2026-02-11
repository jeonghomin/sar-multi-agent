#!/bin/bash
# InSAR API 서버 시작 스크립트

echo "🚀 Starting InSAR Processing API (rag conda env)..."
echo "📡 Server will be available at: http://localhost:8002"
echo "📖 API docs: http://localhost:8002/docs"
echo ""

# insar_processing 폴더로 이동
cd "$(dirname "$0")"

# rag conda 환경에서 실행 (esa_snappy 설치됨)
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate rag
python insar_api.py
