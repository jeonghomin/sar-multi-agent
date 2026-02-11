"""RAG 체인, Query Rewriter, Web Search 도구"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.tools.tavily import TavilySearch
from langsmith import Client
from .llm_config import llm


# ===== RAG Chain =====
rag_system = """당신은 질문에 답변하는 어시스턴트입니다. 
제공된 컨텍스트를 사용하여 질문에 답변하세요.

**중요 지침:**
1. **여러 지역이 있으면 반드시 리스트 형태로 출력하세요.**
   - 각 지역을 번호(1., 2., 3.) 또는 불릿(-)으로 구분
   - 한 문장에 여러 지역을 섞지 마세요

2. **각 지역마다 다음 정보를 포함하세요:**
   - 구체적인 지역명 (원문 그대로)
   - 날짜 (YYYY-MM-DD 또는 "MM월 DD일")
   - 규모 (M 또는 규모)
   - 추가 정보 (좌표, 깊이 등)

3. **출력 형식 예시:**

**나쁜 예** (한 문장에 섞음):
2023년 지진은 필리핀 비콜 지방과 튀르키예에서 발생했으며 규모는 각각 6.1과 7.8이었다.

**좋은 예** (리스트로 구분):
2023년 지진 발생 지역:

1. **필리핀 비콜 지방 우손 북북동쪽 10 km 해역**
   - 날짜: 2023년 11월 15일
   - 규모: M6.1

2. **튀르키예 가지안테프주 누르다으**
   - 날짜: 2023년 2월 6일
   - 규모: M7.8
   - 좌표: 37.225°N 37.021°E

4. 답변이 컨텍스트에 없으면 "모르겠습니다"라고 답하세요.
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
