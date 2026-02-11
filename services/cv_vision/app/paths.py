import os
from pathlib import Path

# 환경변수 설정
# NAS 권한 없으면 로컬 경로 사용
os.environ.setdefault("NAS_ROOT", "/home/mjh/Project/LLM/RAG/ai-service/output")
os.environ.setdefault("MODEL_WEIGHTS_DIR", "/home/mjh/Project/LLM/RAG/ai-service/weights")

# 경로 설정
NAS_ROOT = Path(os.getenv("NAS_ROOT"))
MODEL_WEIGHTS_DIR = Path(os.getenv("MODEL_WEIGHTS_DIR"))

def job_dir(job_id: str) -> Path:
    d = NAS_ROOT / "jobs" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def to_public_uri(abs_path: str) -> str:
    
    return str(abs_path).replace(str(NAS_ROOT), "/files")