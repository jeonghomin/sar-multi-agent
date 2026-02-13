"""환경 설정 및 로깅 설정"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_teddynote import logging

# 환경 변수 로드
load_dotenv(override=True)

# LangSmith 추적 시작
logging.langsmith("rag-vision-agent")

# SAR 데이터 저장 경로 (쉼표로 구분된 여러 경로)
SAR_DATA_PATHS_STR = os.getenv("SAR_DATA_PATHS", "/mnt/sar,/home/mjh/sar_data,/data/sar")
SAR_DATA_PATHS = [Path(p.strip()) for p in SAR_DATA_PATHS_STR.split(",") if p.strip()]

# 기본 SAR 다운로드 경로 (첫 번째 경로 사용)
DEFAULT_SAR_PATH = SAR_DATA_PATHS[0] if SAR_DATA_PATHS else Path("/mnt/sar")
