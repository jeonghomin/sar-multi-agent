# 🛰️ Lumir-X: SAR 위성 데이터 분석 AI 어시스턴트

> LangGraph 기반 멀티 에이전트 시스템으로 자연어를 통한 SAR 위성 데이터 검색, 다운로드, InSAR 분석을 자동화

---

## 📌 프로젝트 개요

**Lumir-X**는 SAR(Synthetic Aperture Radar) 위성 데이터 분석을 위한 대화형 AI 어시스턴트입니다. 사용자가 자연어로 질문하면 자동으로 데이터를 검색하고, 다운로드하며, InSAR 간섭무늬 분석까지 수행합니다.

### 핵심 가치

- **자연어 인터페이스**: 복잡한 CLI 명령어 대신 "터키 지진 InSAR 분석해줘"로 모든 작업 수행
- **완전 자동화**: 데이터 검색 → 다운로드 → 전처리 → 분석까지 원스톱
- **멀티 에이전트**: 각 도메인별 전문 에이전트가 협업하여 복잡한 작업 처리

---

## 🎯 주요 기능

### 1. **InSAR 지표변형 분석** 🌍
지진, 화산 등의 재난으로 인한 지표 변형을 자동 분석
- 간섭무늬(Interferogram) 생성
- 위상(Phase) 분석 및 변형량 계산
- Google Earth 매핑 결과 제공

**예시:**
```
사용자: "튀르키예 가지안테프주 2023년 2월 6일 지진 InSAR 분석해줘"
→ 자동으로 Sentinel-1 데이터 검색
→ 발생 전/후 가장 가까운 데이터 2개 선택
→ ASF에서 다운로드 (총 ~8GB)
→ InSAR 처리 실행
→ Google Earth 링크 제공
```

### 2. **SAR 데이터 자동 검색/다운로드** 📡
Alaska Satellite Facility (ASF) API 연동
- 지역명 → 좌표 자동 변환 (Nominatim)
- 이벤트 날짜 기준 ±2년 범위 검색
- 발생 직전/직후 데이터 우선 정렬
- 다중 선택 다운로드 지원 (1,2,3 형식)

**예시:**
```
사용자: "2023년 2월 터키 지역 데이터 가져와줘"
→ 웹 검색으로 정확한 이벤트 날짜 추출 (2023-02-06)
→ 지역명 → 좌표 변환
→ ASF에서 500개 검색
→ 발생 전 5개 + 후 5개 표시
→ 사용자 선택 후 다운로드
```

### 3. **Vision AI 분석** 🖼️
SAR 이미지에 대한 컴퓨터 비전 분석
- **Segmentation**: LULC(토지 피복) 분류
- **Classification**: 이미지 카테고리 분류
- **Detection**: 객체 탐지 (선박, 차량, 건물)

### 4. **지능형 정보 검색** 🔎
- 웹 검색 (Tavily API) + RAG (FAISS)
- 대화 컨텍스트 기반 Intent 분류
- Chain of Thoughts 기반 추론

---

## 🏗️ 시스템 아키텍처

### Multi-Agent Architecture

```
                    사용자 질문
                        ↓
              ┌─────────────────────┐
              │   Main Router (LLM)  │
              └─────────────────────┘
                        ↓
        ┌───────┬──────┴──────┬──────────┐
        ↓       ↓             ↓          ↓
    Casual   Retrieval    Vision    SAR Processing
    Chat     Agent        Agent      Agent
```

### Retrieval Agent 내부 워크플로우

```
Web Search
    ↓
Intent 분류 (CoT)
    ↓
┌─────────┬─────────────┬───────────────┐
│         │             │               │
Q&A    Location    Data Download
(RAG)   Search      (ASF API)
```

### State Management

```
GraphState (TypedDict)
├─ messages: 대화 히스토리
├─ summary: 과거 대화 요약
├─ intent: 현재 의도 (qa/sar_get_data/sar_search_location)
├─ coordinates: 지역 좌표
├─ awaiting_*: 선택 대기 플래그
└─ sar_result: 분석 결과
```

---

## 🛠️ 기술 스택

### AI/ML Framework
- **LangGraph**: 멀티 에이전트 워크플로우 구현
- **LangChain**: RAG, Structured Output, 체인 구성
- **Ollama**: 로컬 LLM 실행 (Llama 3.1)

### 데이터 & API
- **ASF (Alaska Satellite Facility)**: Sentinel-1 SAR 데이터 다운로드
- **Tavily API**: 실시간 웹 검색
- **Nominatim**: 지역명 → 좌표 변환
- **FAISS**: 벡터 데이터베이스 (SAR 메타데이터)

### Backend
- **FastAPI**: SAR 다운로드 전용 API 서버
- **LangServe**: LangGraph REST API 배포
- **Streamlit**: 웹 UI

### 라이브러리
- **asf_search**: ASF Python SDK
- **Pydantic**: 데이터 검증 및 Structured Output

---

## 💡 핵심 구현 기술

### 1. **Intent-Based Routing (Chain of Thoughts)**

```python
# 3단계 추론으로 정확한 의도 파악
1단계: 핵심 키워드 추출
2단계: 사용자 행동 파악
3단계: 최종 의도 분류
→ qa / sar_get_data / sar_search_location
```

**효과**: 애매한 질문("가지안테프주", "데이터 가져와줘")도 컨텍스트 기반 정확한 분류

### 2. **Stateful Conversation Management**

```python
# 클라이언트 측 State 관리
st.session_state.last_state = result
input_data = {
    "intent": current_state.get("intent"),  # Intent 유지
    "awaiting_*": current_state.get("awaiting_*"),  # 대기 플래그 유지
    ...
}
```

**효과**: LangServe 환경에서도 멀티턴 대화 및 2단계 플로우 구현

### 3. **Event-Centric Data Sorting**

```python
# 이벤트 날짜 기준 전/후 분리 정렬
before_products.sort(reverse=True)  # 발생 직전 우선
after_products.sort()               # 발생 직후 우선
→ 전 5개 + 후 5개 = 총 10개 표시
```

**효과**: InSAR 분석에 최적인 발생 전/후 가장 가까운 데이터 자동 선택

### 4. **2-Stage Download Flow**

```python
# 1단계: 검색 → 리스트 표시
awaiting_single_sar_selection = True
sar_search_results = {...}

# 2단계: 사용자 선택 → 다운로드
selected_indices = [1, 2, 3]  # 다중 선택 지원
→ 각각 순차 다운로드
```

**효과**: 데이터가 많을 때 선택적 다운로드로 시간/비용 절약

### 5. **Summary + Sliding Window**

```python
# 대화가 길어지면 자동 요약
if len(messages) > 8:
    summary = llm_summary.invoke(old_messages)
    messages = messages[-10:]  # 최근 10개만 유지
```

**효과**: 무제한 대화 가능, 토큰 절약, 장기 컨텍스트 유지

---

## 📊 프로젝트 규모

| 항목 | 내용 |
|------|------|
| **전체 코드** | ~5,000줄 |
| **핵심 모듈** | 8개 |
| **LangGraph 노드** | 15개 |
| **API 엔드포인트** | 4개 |
| **에이전트** | 4개 (Casual, Retrieval, Vision, SAR) |
| **Intent 타입** | 3개 (qa, sar_get_data, sar_search_location) |
| **State 필드** | 20개+ |

---

## 🎨 핵심 파일 구조

```
agent_cv/
├── graph.py                    # LangGraph 워크플로우 (452줄)
├── state.py                    # GraphState 정의 (51줄)
├── nodes/
│   ├── __init__.py            # Main Router (173줄)
│   ├── retrieval/
│   │   ├── search_nodes.py    # Web Search + Intent 분류 (673줄)
│   │   ├── download_node.py   # SAR 다운로드 (1,070줄)
│   │   └── generation_nodes.py # 답변 생성
│   ├── sar/
│   │   └── insar_processing.py # InSAR 처리
│   └── vision/
│       └── vision_nodes.py     # 이미지 분석
├── core/
│   ├── llm_config.py          # LLM 설정
│   └── chains.py              # RAG 체인
├── routing/
│   └── routers.py             # 라우팅 룰
├── sar_api/
│   ├── sar_download_api.py    # FastAPI 서버
│   └── sar_download_utils.py  # ASF SDK 래퍼
└── web_ui.py                  # Streamlit UI (153줄)
```

---

## 🚀 주요 성과

### 1. **복잡한 워크플로우 자동화**
- 수동: 50+ 단계 (좌표 찾기 → API 호출 → 데이터 선택 → 다운로드 → 전처리 → 분석)
- 자동: 1단계 (자연어 질문)

### 2. **지능형 라우팅 시스템**
- Chain of Thoughts 기반 Intent 분류
- 컨텍스트 기반 Previous Intent 유지
- 의문사 우선 감지로 Q&A 정확도 향상

### 3. **Event-Centric 데이터 처리**
- 절대값 정렬 → 전/후 분리 정렬로 개선
- 이벤트 날짜 기준 최적 데이터 자동 선택
- ±2년 범위 검색으로 데이터 커버리지 확대

### 4. **확장 가능한 아키텍처**
- 모듈화된 노드 구조 (새 노드 추가 용이)
- 플러그인 방식 에이전트 (새 에이전트 추가 용이)
- API 기반 분리 (SAR 다운로드 서버 독립 실행)

---

## 💼 포트폴리오 강조 포인트

### 기술적 도전과제 해결

#### 1. **LangServe 환경에서의 State 관리**
**문제**: LangServe가 TypedDict의 Optional 필드를 무시하고 422 에러 발생
**해결**: 클라이언트 측 명시적 State 관리 (`st.session_state.last_state`)

#### 2. **복잡한 멀티턴 대화 관리**
**문제**: 데이터 선택 대기 중 새로운 질문 구분 필요
**해결**: 의문사 우선 감지 + Previous Intent + Awaiting 플래그 3단계 체크

#### 3. **ASF API maxResults 제한**
**문제**: 최신 50개만 반환되어 과거 데이터 누락
**해결**: maxResults를 500개로 확대하여 전체 기간 커버

#### 4. **날짜 파싱 오류**
**문제**: "2023년 2월 6일"을 인덱스 [2, 6, 2023]으로 오인
**해결**: 날짜 패턴 제거 + "X번" 형식 우선 파싱

---

## 📈 기술 성장 포인트

### 학습한 개념
- ✅ LangGraph의 복잡한 State 관리
- ✅ Structured Output (Pydantic) 활용
- ✅ Chain of Thoughts 프롬프팅
- ✅ 멀티 에이전트 협업 패턴
- ✅ FastAPI + LangServe 통합
- ✅ ASF API 및 SAR 데이터 처리

### 구현 패턴
- ✅ Intent-Based Routing
- ✅ 2-Stage Flow (검색 → 선택)
- ✅ Event-Centric Sorting
- ✅ Client-Side State Management
- ✅ Summary + Sliding Window

---

## 🎬 데모 시나리오

### 시나리오 1: 간단한 InSAR 분석
```
👤 "안녕"
🤖 "안녕하세요! SAR 데이터 분석을 도와드릴게요..."

👤 "터키 2023년 2월 6일 지진 InSAR 분석해줘"
🤖 "튀르키예 가지안테프주의 SAR 데이터를 검색했습니다.
    발생 전 5개, 후 5개 총 10개 표시
    
    2023-02-05 (📌 발생 1일 전)
      [0] S1A_IW_SLC__...
    2023-02-07 (📌 발생 1일 후)
      [5] S1A_IW_SLC__...
    
    Master와 Slave를 선택해주세요"

👤 "Master 0, Slave 5"
🤖 "다운로드 중... (8.2GB)
    InSAR 처리 시작...
    
    ✅ InSAR 처리 완료!
    📊 최대 변형량: 2.5m
    🌍 Google Earth: [링크]"
```

### 시나리오 2: 정보가 부족한 경우
```
👤 "터키 데이터 가져와줘"
🤖 "지역명이 너무 포괄적입니다.
    더 구체적인 지역명을 알려주세요"

👤 "가지안테프주"
🤖 (Previous Intent 유지 → retrieval)
    "가지안테프주의 데이터를 검색합니다..."
```

### 시나리오 3: Intent 전환
```
👤 "일본 데이터 가져와줘"
🤖 [리스트 표시]

👤 "혹시 다른 지역 어디 있어?"
🤖 (Q&A 의문사 감지 → Intent 전환)
    "2023년 주요 지진 발생 지역은..."
```

---

## 🔧 개발 환경

### 실행 방법
```bash
# 1. SAR 다운로드 API 시작
cd sar_api
python sar_download_api.py  # localhost:8001

# 2. 메인 서버 시작
python server.py            # localhost:8000

# 3. Streamlit UI
streamlit run web_ui.py     # localhost:8501
```

### 배포 옵션
- **LangServe**: REST API 서버 (프로덕션)
- **LangGraph Studio**: 개발/디버깅 환경
- **Streamlit**: 웹 UI 프로토타입

---

## 📚 학습 자료 및 참고

### 사용된 기술
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [ASF Data Search](https://search.asf.alaska.edu/)
- [Sentinel-1 SAR User Guide](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar)

### 프로젝트 배경
- InSAR 분석은 전문가도 수동으로 수행 시 수 시간 소요
- 비전문가는 CLI 명령어와 복잡한 파라미터로 진입 장벽이 높음
- AI 에이전트로 자동화하여 접근성 향상

---

## 🎓 향후 개선 계획

### 단기
- [ ] 자동 다운로드 (이벤트 날짜 명확 시)
- [ ] Buffer 동적 조정 (데이터 없으면 범위 확대)
- [ ] 다른 SAR 위성 지원 (ALOS-2, COSMO-SkyMed)

### 장기
- [ ] InSAR 결과 시각화 개선
- [ ] 시계열 분석 (여러 시점 비교)
- [ ] Change Detection 자동화
- [ ] 클라우드 배포 (AWS/GCP)

---

## 📞 기술 스택 요약

**Backend**: Python, FastAPI, LangGraph, LangChain  
**AI/ML**: Ollama (Llama 3.1), Chain of Thoughts, RAG  
**Data**: ASF API, FAISS, Nominatim  
**Frontend**: Streamlit  
**Deployment**: LangServe, Docker-ready  

---

## 🏆 프로젝트 의의

이 프로젝트는 **복잡한 도메인 지식(SAR 데이터 분석)과 최신 AI 기술(LangGraph, CoT)을 결합**하여, 전문가가 아니어도 자연어만으로 고급 위성 데이터 분석을 수행할 수 있도록 만든 **실용적인 AI 에이전트 시스템**입니다.

특히 **멀티 에이전트 협업**, **컨텍스트 기반 State 관리**, **Event-Centric 데이터 처리**는 실제 프로덕션 환경에서도 활용 가능한 패턴입니다.

---

**개발 기간**: 2024-2026  
**개발 인원**: 1명  
**GitHub**: [링크]  
**데모**: [링크]
