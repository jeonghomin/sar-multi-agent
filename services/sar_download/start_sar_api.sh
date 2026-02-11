#!/bin/bash
# SAR Download API 서버 시작 스크립트

echo "🚀 Starting SAR Download API Server..."
echo "📡 Server will be available at: http://localhost:8001"
echo "📖 API docs: http://localhost:8001/docs"
echo ""

# sar_api 폴더로 이동
cd "$(dirname "$0")"

# Python 모듈로 실행
python sar_download_api.py
