"""
SAR Download API 서버 실행
python -m sar_api 로 실행
"""
import uvicorn
from .sar_download_api import app

if __name__ == "__main__":
    print("🚀 Starting SAR Download API Server...")
    print("📡 Server will be available at: http://localhost:8001")
    print("📖 API docs: http://localhost:8001/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
