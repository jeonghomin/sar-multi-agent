# Agent CV 프로젝트 구조

## 📁 폴더 구조

```
agent_cv/
├── core/                      # 핵심 모듈 (모델, LLM 설정, 체인)
│   ├── models.py             # Pydantic 모델 정의
│   ├── llm_config.py         # LLM 인스턴스 및 Structured Output 설정
│   ├── chains.py             # RAG 체인, Query Rewriter, Web Search 도구
│   └── __init__.py
│
├── routing/                   # 라우팅 모듈
│   ├── routers.py            # Main, Vision, Retrieval, SAR 라우터
│   └── __init__.py
│
├── evaluation/                # 평가 모듈
│   ├── graders.py            # 문서, 환각, 답변 평가기
│   └── __init__.py
│
├── nodes/                     # 노드 함수들
│   ├── retrieval/            # Retrieval Agent 노드
│   │   ├── retrieval_nodes.py
│   │   └── __init__.py
│   ├── vision/               # Vision Agent 노드
│   │   ├── vision_nodes.py
│   │   └── __init__.py
│   ├── sar/                  # SAR Processing Agent 노드
│   │   ├── sar_nodes.py
│   │   └── __init__.py
│   └── __init__.py
│
├── nodes.py                   # 통합 노드 (모든 노드 export)
├── graph.py                   # LangGraph 워크플로우 정의
├── state.py                   # Graph State 정의
├── config.py                  # 설정 파일
├── insar_datasets.py         # InSAR 데이터셋 설정
├── pdf_setup.py              # PDF/VectorStore 설정
├── location_utils.py         # 좌표 변환 유틸리티
├── vision_tools.py           # Vision AI 도구
├── sar_segmentation_node.py  # SAR 분할 노드
└── main.py                   # 메인 실행 파일
```

## 📦 모듈 설명

### 1. **core/** - 핵심 모듈
프로젝트의 기본 구성 요소들

- **models.py**: Pydantic 모델 정의
  - `RouterQuery`: Main 라우팅
  - `RouteQuery`: Retrieval 라우팅
  - `GradeDocument`, `GradeHallucination`, `GradeAnswer`: 평가 모델
  - `VisionAgent`, `SARProcessingAgent`: 태스크 라우팅 모델

- **llm_config.py**: LLM 설정 및 Structured Output
  - `llm`, `llm_vision`: ChatOllama 인스턴스
  - Structured LLM 인스턴스들 (router, retrieval, vision, sar, grader 등)

- **chains.py**: 체인 및 도구
  - `rag_chain`: RAG 체인
  - `query_rewriter`: 쿼리 재작성
  - `web_search_tool`: Tavily 웹 검색

### 2. **routing/** - 라우팅 모듈
질문을 적절한 에이전트로 라우팅

- **routers.py**:
  - `main_agent`: Vision vs Retrieval vs SAR Processing 선택
  - `vision_router`: Segmentation vs Classification vs Detection
  - `question_router`: VectorStore vs Web Search vs Extract Coordinates
  - `sar_router`: InSAR vs Change Detection vs Analysis

### 3. **evaluation/** - 평가 모듈
문서 및 생성 결과 평가

- **graders.py**:
  - `retrieval_grader`: 문서 관련성 평가
  - `hallucination_grader`: 환각 감지
  - `answer_grader`: 답변 품질 평가

### 4. **nodes/** - 노드 함수들
LangGraph의 실제 노드 구현

#### 4.1. **retrieval/** - Retrieval Agent
- `route_question`: 질문 라우팅 (vectorstore/web_search/extract_coordinates)
- `web_search`: 웹 검색 수행 및 지역명 추출
- `extract_coordinates`: 지역명 → 좌표 변환
- `retrieve`: SAR 메타데이터 검색 (vectorstore)
- `grade_document`: 문서 평가
- `generate`: 최종 답변 생성
- `grade_hallucination`: 환각 감지 및 답변 품질 평가
- `rewrite`: 쿼리 재작성

#### 4.2. **vision/** - Vision Agent
- `vision_task_router`: Vision 태스크 선택
- `run_segmentation`: 이미지 분할
- `run_classification`: 이미지 분류
- `run_detection`: 객체 탐지
- `vision_generate`: Vision 결과 생성

#### 4.3. **sar/** - SAR Processing Agent
- `run_insar`: InSAR 처리 (지표 변형 분석)
- `sar_generate`: SAR 결과 생성

## 🔄 Import 구조

### 외부에서 사용
```python
from nodes import (
    # 모든 노드, 라우터, 평가기, 체인 등을 import 가능
    main_router,
    route_question,
    vision_task_router,
    sar_task_router,
    # ...
)
```

### 내부 모듈 간 의존성
```
core (models, llm_config, chains)
  ↓
routing (routers) ← core.llm_config
evaluation (graders) ← core.llm_config
  ↓
nodes/ ← routing, evaluation, core
```

## 🚀 장점

1. **모듈화**: 각 기능이 명확하게 분리되어 유지보수 용이
2. **확장성**: 새로운 노드나 라우터 추가가 쉬움
3. **가독성**: 폴더 구조만 봐도 프로젝트 전체 파악 가능
4. **재사용성**: 각 모듈을 독립적으로 테스트 및 재사용 가능
5. **import 단순화**: `nodes.py`를 통해 모든 것을 한 곳에서 import

## 📝 개발 가이드

### 새로운 노드 추가하기
1. 적절한 폴더에 노드 함수 작성 (예: `nodes/retrieval/new_node.py`)
2. 해당 폴더의 `__init__.py`에 export 추가
3. `nodes/__init__.py`에 import 추가
4. `nodes.py`에 export 추가
5. `graph.py`에서 노드 연결

### 새로운 라우터 추가하기
1. `routing/routers.py`에 라우터 정의
2. `routing/__init__.py`에 export 추가
3. `nodes.py`에서 import하여 export

### 새로운 평가기 추가하기
1. `evaluation/graders.py`에 평가기 정의
2. `evaluation/__init__.py`에 export 추가
3. `nodes.py`에서 import하여 export
