# Agent CV - LangServe 배포 가이드

Multi-agent system for SAR image processing, computer vision, and RAG를 LangServe로 배포하는 가이드입니다.

## 📦 설치

### Poetry 사용 (권장)

```bash
# rag 환경 활성화
conda activate rag

# 프로젝트 디렉토리로 이동
cd /home/mjh/Project/LLM/RAG/rag-study/agent_cv

# LangServe 설치 (이미 완료)
poetry add "langserve[all]"
```

### 의존성 확인

```bash
poetry show langserve
```

## 🚀 서버 실행

### 방법 1: Poetry로 실행

```bash
conda activate rag
cd /home/mjh/Project/LLM/RAG/rag-study/agent_cv
poetry run python server.py
```

### 방법 2: 직접 실행

```bash
conda activate rag
cd /home/mjh/Project/LLM/RAG/rag-study/agent_cv
python server.py
```

### 방법 3: Uvicorn으로 직접 실행

```bash
conda activate rag
cd /home/mjh/Project/LLM/RAG/rag-study/agent_cv
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

## 🌐 API 엔드포인트

서버가 실행되면 다음 URL에서 접근할 수 있습니다:

### 웹 인터페이스
- **API 문서 (Swagger)**: http://localhost:8000/docs
- **API 문서 (ReDoc)**: http://localhost:8000/redoc
- **Playground**: http://localhost:8000/agent_cv/playground/

### API 엔드포인트
- **Root**: http://localhost:8000/
- **Health Check**: http://localhost:8000/health
- **Invoke**: POST http://localhost:8000/agent_cv/invoke
- **Batch**: POST http://localhost:8000/agent_cv/batch
- **Stream**: POST http://localhost:8000/agent_cv/stream
- **Stream Log**: POST http://localhost:8000/agent_cv/stream_log
- **Stream Events**: POST http://localhost:8000/agent_cv/stream_events

## 💻 클라이언트 사용 예제

### Python 클라이언트

```bash
# 다른 터미널에서 실행
conda activate rag
cd /home/mjh/Project/LLM/RAG/rag-study/agent_cv
python client_example.py
```

### 코드 예제

```python
from langserve import RemoteRunnable

# 원격 서버 연결
remote_graph = RemoteRunnable("http://localhost:8000/agent_cv/")

# RAG 쿼리
result = remote_graph.invoke({
    "question": "군 복무 기간은 얼마인가요?",
    "messages": [],
    "documents": [],
    "generation": "",
})

print(result["generation"])
```

### cURL 예제

```bash
# Invoke 엔드포인트 호출
curl -X POST "http://localhost:8000/agent_cv/invoke" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "question": "군 복무 기간은 얼마인가요?",
      "messages": [],
      "documents": [],
      "generation": ""
    },
    "config": {
      "configurable": {
        "thread_id": "test-thread-1"
      }
    }
  }'
```

## 🎯 주요 기능

### 1. Multi-Agent 지원
- **Retrieval Agent**: RAG 기반 문서 검색 및 답변
- **Vision Agent**: SAR 이미지 분석 (분류, 탐지, 세그멘테이션)
- **SAR Processing Agent**: InSAR 처리

### 2. 스트리밍 지원
- `/stream`: 결과 스트리밍
- `/stream_log`: 중간 단계 포함 스트리밍
- `/stream_events`: 모든 이벤트 스트리밍

### 3. 상태 관리
- MemorySaver를 사용한 대화 상태 저장
- thread_id로 세션 관리

## 🔧 설정 옵션

### 포트 변경

`server.py` 파일에서:

```python
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8100,  # 원하는 포트로 변경
        log_level="info",
    )
```

### CORS 설정

프로덕션 환경에서는 특정 도메인만 허용:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # 특정 도메인만
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📊 모니터링

### LangSmith 통합

환경 변수 설정:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="your-api-key"
export LANGCHAIN_PROJECT="agent-cv"
```

## 🐛 문제 해결

### 의존성 충돌

```bash
# langflow, crewai와 충돌하는 경우
pip uninstall langflow crewai -y

# poetry로 재설치
poetry install
```

### 포트 이미 사용 중

```bash
# 포트 사용 확인
lsof -i :8000

# 프로세스 종료
kill -9 <PID>
```

### Poetry 관련 문제

```bash
# 캐시 정리
poetry cache clear pypi --all

# 가상환경 재생성
poetry env remove python
poetry install
```

## 🚢 프로덕션 배포

### Gunicorn 사용

```bash
conda activate rag
gunicorn server:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300
```

### Docker (선택사항)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Poetry 설치
RUN pip install poetry

# 의존성 복사 및 설치
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-dev

# 코드 복사
COPY . .

# 서버 실행
CMD ["poetry", "run", "python", "server.py"]
```

## 📚 참고 자료

- [LangServe 공식 문서](https://python.langchain.com/docs/langserve)
- [LangGraph Cloud](https://langchain-ai.github.io/langgraph/cloud/)
- [FastAPI 문서](https://fastapi.tiangolo.com/)

## ⚠️ 주의사항

- LangServe는 간단한 Runnable 배포에 최적화되어 있습니다
- 복잡한 LangGraph 애플리케이션은 **LangGraph Cloud**를 권장합니다
- 프로덕션 환경에서는 보안 설정 (HTTPS, 인증 등)을 추가하세요
