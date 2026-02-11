# 🛰️ SAR Multi-Agent System

LangGraph 기반 멀티 에이전트 시스템 - SAR 데이터 검색, 다운로드, InSAR 처리를 위한 통합 AI 어시스턴트

> Intelligent SAR Data Processing with LangGraph, SNAP, and FastAPI

## 🌟 주요 기능

### 1. Retrieval Agent
- 웹 검색 및 정보 추출
- 지역명 → 좌표 변환
- 날짜/이벤트 정보 추출
- RAG 기반 Q&A

### 2. SAR Processing Agent
- **SAR 데이터 검색**: 위치/날짜 기반 Sentinel-1 데이터 검색
- **자동 다운로드**: ASF API 연동
- **InSAR 처리**: SNAP을 사용한 지표변형 분석
  - Master/Slave 자동 선택
  - 백그라운드 처리 (20-30분)
  - Phase/Coherence map 생성

### 3. Vision Agent
- 이미지 분할 (Segmentation)
- 객체 탐지 (Detection)
- 이미지 분류 (Classification)

## 🏗️ 프로젝트 구조

```
sar-multi-agent/
├── server.py                   # Agent 메인 서버 (port 8000)
├── web_ui.py                   # Streamlit UI
├── graph.py                    # LangGraph 워크플로우
├── state.py                    # GraphState 정의
│
├── nodes/                      # Agent 노드들
│   ├── retrieval/              # 검색/다운로드 노드
│   │   └── prompts/            # LLM 프롬프트
│   ├── sar/                    # SAR/InSAR 노드
│   │   └── prompts/
│   └── vision/                 # 비전 노드
│       └── prompts/
│
├── services/                   # 외부 API 서비스
│   ├── sar_download/           # SAR 다운로드 API (port 8001)
│   ├── insar_processing/       # InSAR 처리 API (port 8002)
│   └── cv_vision/              # Computer Vision API (예정)
│
└── scripts/                    # 유틸리티 스크립트
    ├── start_all.sh            # 모든 서비스 시작
    ├── stop_all.sh             # 모든 서비스 종료
    └── check_services.sh       # 서비스 상태 확인
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# Conda 환경 생성
conda create -n rag python=3.11
conda activate rag

# Python 패키지 설치
pip install -r requirements.txt

# SNAP Python API 설치 (InSAR용)
cd services/insar_processing
bash INSTALL_ESA_SNAPPY.sh
cd ../..
```

### 2. 서비스 시작

**Option A: 모든 서비스 한 번에 시작**
```bash
bash scripts/start_all.sh
```

**Option B: 개별 서비스 시작**
```bash
# SAR Download API (port 8001)
cd services/sar_download
bash start_sar_api.sh

# InSAR Processing API (port 8002)
cd services/insar_processing
bash start_insar_api.sh

# Agent Server (port 8000)
python server.py
```

### 3. UI 접속

**Streamlit UI:**
```bash
streamlit run web_ui.py
```

**LangServe Playground:**
```
http://localhost:8000/chat/playground
```

## 📡 API 포트

| 서비스 | 포트 | 용도 |
|--------|------|------|
| Agent Server | 8000 | LangGraph 메인 에이전트 |
| SAR Download | 8001 | Sentinel-1 데이터 검색/다운로드 |
| InSAR Processing | 8002 | SNAP InSAR 처리 |
| CV Vision | 8003 | Computer Vision 처리 (예정) |

## 🧪 사용 예시

### InSAR 처리
```
사용자: "2023년 터키 지진 데이터로 InSAR 해줘"
Agent: → 위치 검색 → 데이터 다운로드 → Master/Slave 선택 → InSAR 처리
```

### 직접 파일 지정
```
사용자: "/mnt/sar/S1A_...zip /mnt/sar/S1A_...zip 이 파일들로 InSAR 해줘"
Agent: → Master/Slave 선택 → 파라미터 입력 → InSAR 처리 시작
```

### SAR 데이터 검색
```
사용자: "2024년 일본 노토반도 지진 데이터 가져와줘"
Agent: → 위치 검색 → 좌표 변환 → SAR 데이터 검색 → 다운로드
```

## 🛠️ 개발

### 프롬프트 수정
프롬프트는 별도 파일로 관리됩니다:
- `nodes/retrieval/prompts/` - 검색/분류 프롬프트
- `nodes/sar/prompts/` - SAR/InSAR 프롬프트

### 코드 구조
- **LangGraph**: 워크플로우 그래프 정의 (`graph.py`)
- **State 관리**: TypedDict 기반 (`state.py`)
- **노드**: 각 처리 단계별 함수 (`nodes/`)
- **라우팅**: 조건부 엣지 (`graph.py`)

## 📊 아키텍처

```
User → Streamlit UI → Agent Server (LangGraph)
                         ↓
                   [Main Router]
                    /    |    \
                   /     |     \
            Retrieval  SAR    Vision
               ↓       ↓         ↓
          Web Search  SAR API  CV API
               ↓       ↓
          RAG/QA   InSAR API
```

## 🔧 문제 해결

### 서비스가 시작되지 않을 때
```bash
# 서비스 상태 확인
bash scripts/check_services.sh

# 기존 프로세스 종료
bash scripts/stop_all.sh

# 다시 시작
bash scripts/start_all.sh
```

### InSAR 처리 오류
- SNAP 설치 확인: `/home/mjh/esa-snap`
- esa_snappy 설정 확인: `services/insar_processing/INSTALL_ESA_SNAPPY.sh`
- 충분한 디스크 공간 (10GB+)

## 📦 Dependencies

```bash
# Core
langgraph==0.2.45
langchain==0.3.7
langchain-community==0.3.5
langchain-openai==0.2.5

# API
fastapi==0.115.5
uvicorn==0.32.1

# SAR Processing
asf_search==9.0.3
esa_snappy (SNAP Python API)

# Utilities
requests==2.32.3
pydantic==2.10.2
```

## 📝 License

MIT

## 👥 Contributors

- Minjeong Ha (mjh)
