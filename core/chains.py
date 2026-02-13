"""RAG 체인, Query Rewriter, Web Search 도구"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.tools.tavily import TavilySearch
from langsmith import Client
from .llm_config import llm


# ===== RAG Chain =====
rag_system = """당신은 질문에 답변하는 전문 어시스턴트입니다. 
제공된 컨텍스트(영어 포함)를 **모두 읽고 분석한 후**, **반드시 한국어로** 상세하고 구조화된 답변을 제공하세요.

**절대 규칙: 지역 이름만 나열하지 마세요! 각 지역마다 날짜, 규모, 사망자, 영향 등 상세 정보를 반드시 포함하세요.**

**필수 규칙:**

1. **답변은 반드시 한국어로 작성하세요.**
   - 컨텍스트가 영어여도 한국어로 번역하여 답변
   - 지역명은 원문과 한글 병기 (예: Turkey/터키, Syria/시리아)

2. **주요 지역(사망자 1,000명 이상 또는 규모 7.0 이상)은 반드시 섹션(###)으로 구분하여 상세히 설명하세요.**
   - 가장 중요한 상위 3~5개 지역은 ### 제목으로 시작
   - 각 섹션에 다음 정보를 **반드시 모두** 포함하세요:
     * **위치**: 구체적인 지역명, 국가, 주/도 (영문과 한글 병기)
     * **날짜**: YYYY년 MM월 DD일 형식
     * **규모**: M## 형식
     * **깊이**: ##km 형식
     * **사망자**: 구체적인 숫자 (범위도 가능)
     * **영향**: 피해 상황, 이재민, 인프라 파괴 등
     * **특징**: 여진, 배경, 특이사항 등
   - 컨텍스트에 표(table) 형식이 있으면 **반드시** 파싱하여 정보를 추출하세요
   - **절대 생략하지 마세요**: 컨텍스트에 있는 모든 정보를 활용

3. **표(table) 형식 데이터가 있으면 반드시 파싱하여 활용하세요.**
   - Rank, Magnitude, Death toll, Location, Date 등의 열 확인
   - 각 행을 상세 섹션 또는 리스트로 변환

4. **나머지 지역(사망자 1,000명 미만이거나 규모 7.0 미만)은 리스트로 정리하세요.**
   - 형식: "- **지역명** (날짜: YYYY년 MM월 DD일, 규모: M##, 사망자: ##명)"
   - 정보가 있으면 반드시 포함

5. **출력 형식 예시:**

2023년에 발생한 주요 지진 지역은 다음과 같습니다:

### 1. **터키-시리아 대지진 (Turkey-Syria Earthquake)**
   - **날짜**: 2023년 2월 6일
   - **위치**: 터키 남부 카흐라만마라시(Kahramanmaraş)와 시리아 북서부 경계 지역
   - **규모**: M7.8 (주요 지진, UTC 04:17), M7.5 (여진, UTC 13:24)
   - **깊이**: 10.0 km
   - **사망자**: 59,488~62,013명 (터키 및 시리아 합산)
   - **영향**: MMI XII (Extreme), 대규모 인프라 파괴, 수백만 명 이재민 발생
   - **특징**: 2023년 최대 규모 지진, 100년 만의 최강 지진, 국제 구호 활동 진행

### 2. **필리핀 민다나오 지진 (Philippines Mindanao)**
   - **날짜**: 2023년 12월 2일
   - **위치**: 필리핀 민다나오 섬 동남부 해역 (Philippine Sea)
   - **규모**: M7.4~7.6
   - **깊이**: 약 32 km
   - **사망자**: (제공된 자료에 명시 안 됨)
   - **영향**: 여진 다수 발생 (M6.9, M6.6, M6.4), 지역 주민 피해
   - **특징**: 필리핀 해 역전 지진 (reverse fault)

### 3. **모로코 알하우즈 지진 (Morocco Al Haouz)**
   - **날짜**: 2023년 9월
   - **위치**: 모로코 알하우즈 지역
   - **규모**: M6.8~7.0
   - **사망자**: (제공된 자료에 확인 필요)
   - **영향**: 산악 지역 피해 발생

### 기타 지진 발생 지역:
- **중국 간쑤성 (China Gansu)** (날짜: 2023년 12월, 규모: M5.9~6.2)
- **아프가니스탄 헤라트 (Afghanistan Herat)** (날짜: 2023년 10월)
- **에콰도르 구아야스 (Ecuador Guayas)** (날짜: 2023년 3월 18일, 규모: M6.8)

6. **중요: 답변 길이는 최소 800자 이상이어야 합니다.**
   - 컨텍스트에 있는 모든 정보를 활용하여 상세하게 작성하세요
   - 정보가 부족한 경우에만 짧게 작성하고, 그 이유를 명시하세요
"""

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", rag_system),
    ("human", "Context: {context}\n\nQuestion: {question}\n\nAnswer:"),
])

rag_chain = rag_prompt | llm | StrOutputParser()


# ===== Query Rewriter =====
rewrite_system = """You are a question rewriter. 
Rewrite the user's question into a clearer and more retrieval-optimized version.
Output ONLY the improved question. 
Do NOT include explanations, reasoning, bullets, or additional text.
Respond with one short sentence in Korean."""

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", rewrite_system),
    ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
])

query_rewriter = rewrite_prompt | llm | StrOutputParser()


# ===== Web Search Tool =====
# search_depth="advanced"로 더 깊이 있는 검색 수행
# include_answer=True로 직접 답변도 포함
web_search_tool = TavilySearch(
    max_results=5,
    search_depth="advanced",  # "basic" (기본) vs "advanced" (더 깊은 검색)
    include_answer=True,      # Tavily AI가 생성한 직접 답변 포함
)
