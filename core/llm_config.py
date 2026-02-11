"""LLM 인스턴스 및 설정"""
from langchain_ollama.chat_models import ChatOllama
from .models import (
    RouterQuery, 
    RouteQuery, 
    GradeDocument, 
    GradeHallucination, 
    GradeAnswer,
    VisionAgent
)


# LLM 인스턴스
llm = ChatOllama(model="qwen3:14b", temperature=0)
llm_vision = ChatOllama(model="qwen3:14b", temperature=0)
llm_summary = ChatOllama(model="qwen3:14b", temperature=0)  # 요약 전용 (일관성 중요)

# Structured LLM 출력
structured_llm_router = llm.with_structured_output(RouterQuery)
structured_llm_retrieval = llm.with_structured_output(RouteQuery)
structured_llm_vision = llm.with_structured_output(VisionAgent)
structured_llm_grader = llm.with_structured_output(GradeDocument)
structured_llm_hallucination = llm.with_structured_output(GradeHallucination)
structured_llm_answer = llm.with_structured_output(GradeAnswer)
