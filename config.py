"""환경 설정 및 로깅 설정"""
from dotenv import load_dotenv
from langchain_teddynote import logging

# 환경 변수 로드
load_dotenv(override=True)

# LangSmith 추적 시작
logging.langsmith("rag-vision-agent")
