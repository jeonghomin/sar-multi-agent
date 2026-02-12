#!/usr/bin/env python
"""LangServe 서버 - Agent CV Multi-Agent System"""
import config  # Load environment variables (.env file)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langgraph.checkpoint.memory import MemorySaver
from graph import workflow
import uvicorn

# FastAPI 앱 생성
app = FastAPI(
    title="Agent CV API",
    version="1.0.0",
    description="Multi-agent system for SAR image processing, computer vision, and RAG",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 설정 (필요한 경우)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용하도록 변경
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 체크포인터 추가 (상태 저장용)
memory = MemorySaver()
compiled_graph = workflow.compile(checkpointer=memory)

# 기본 thread_id를 자동으로 추가하는 함수
from typing import Dict, Any
from langchain_teddynote.messages import random_uuid

def add_default_thread_id(config: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, Any]:
    """요청마다 기본 thread_id를 자동으로 추가"""
    if "configurable" not in config:
        config["configurable"] = {}
    if "thread_id" not in config["configurable"]:
        # 세션별로 다른 thread_id 생성 (실제로는 세션 관리가 필요)
        config["configurable"]["thread_id"] = "default-session"
    return config

# LangServe 라우트 추가
add_routes(
    app,
    compiled_graph,
    path="/agent_cv",
    enabled_endpoints=["invoke", "batch", "stream", "stream_log", "stream_events"],
    per_req_config_modifier=add_default_thread_id,
)

# 헬스 체크 엔드포인트
@app.get("/")
async def root():
    return {
        "message": "Agent CV API is running",
        "endpoints": {
            "docs": "/docs",
            "playground": "/agent_cv/playground/",
            "invoke": "/agent_cv/invoke",
            "stream": "/agent_cv/stream",
            "stream_events": "/agent_cv/stream_events",
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    # 서버 실행
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
