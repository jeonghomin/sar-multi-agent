"""문서 및 생성 평가기"""
from langchain_core.prompts import ChatPromptTemplate
from core.llm_config import (
    structured_llm_grader,
    structured_llm_hallucination,
    structured_llm_answer
)


# ===== Document Grader =====
grade_system = """ You are a grader assessing relevance of a retrieved document to a user question. \n 
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
"""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", grade_system),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
])

retrieval_grader = grade_prompt | structured_llm_grader


# ===== Hallucination Grader =====
hallucination_system = """You are a grader assessing whether an LLM generation is hallucinated.
Give a binary score 'yes' or 'no' score to indicate whether the generation is supported by the retrieved facts."""

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", hallucination_system),
    ("human", "Sets of facts: \n\n {documents} \n\n LLM generation: {generation}"),
])

hallucination_grader = hallucination_prompt | structured_llm_hallucination


# ===== Answer Grader =====
answer_system = """You are a grader assessing whether an answer addresses / resolves a question
Give a binary score 'yes' or 'no'. 'yes' means that the answer solves the question."""

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", answer_system),
    ("human", "User question: {question} \n\n LLM generation: {generation}"),
])

answer_grader = answer_prompt | structured_llm_answer
