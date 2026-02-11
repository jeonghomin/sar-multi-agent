#!/bin/bash
# esa_snappy 설치 스크립트 (rag 환경)

echo "🔧 Installing esa_snappy to rag conda environment..."

# 1. rag 환경의 Python 경로 확인
RAG_PYTHON=/home/mjh/.conda/envs/rag/bin/python

if [ ! -f "$RAG_PYTHON" ]; then
    echo "❌ rag environment not found: $RAG_PYTHON"
    exit 1
fi

echo "✅ Found rag Python: $RAG_PYTHON"

# 2. SNAP snappy-conf 실행
echo "🔧 Running SNAP snappy-conf..."
/home/mjh/esa-snap/bin/snappy-conf "$RAG_PYTHON"

# 3. 테스트
echo "🧪 Testing esa_snappy import..."
conda run -n rag python -c "from esa_snappy import ProductIO; print('✅ esa_snappy installed successfully!')"

echo ""
echo "✅ Done! Now you can use esa_snappy in 'rag' environment"
echo ""
echo "Usage:"
echo "  conda activate rag"
echo "  python -c \"from esa_snappy import ProductIO; print('OK')\""
