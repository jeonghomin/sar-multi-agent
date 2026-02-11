"""FAISS VectorStore 로드 및 체인 생성"""
import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings

# FAISS VectorStore 경로
vectorstore_path = Path(__file__).parent / "faiss_vectorstore"

# VectorStore 로드 시도
try:
    if vectorstore_path.exists() and (vectorstore_path / "index.faiss").exists():
        print(f"FAISS VectorStore 로드 중: {vectorstore_path}")
        embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")
        vectorstore = FAISS.load_local(
            str(vectorstore_path),
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Retriever 생성
        pdf_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
        
        print(f"FAISS VectorStore 로드 완료!")
        print(f"  - VectorStore 경로: {vectorstore_path}")
        
        # 호환성을 위해 pdf, pdf_chain 변수도 유지 (None으로 설정)
        pdf = None
        pdf_chain = None
        # vectorstore를 export하여 nodes.py에서 접근 가능하도록
        # (vectorstore 변수는 이미 위에서 생성됨)
    else:
        print(f"경고: FAISS VectorStore를 찾을 수 없습니다: {vectorstore_path}")
        print("VectorStore 파일이 없어 기본값을 사용합니다.")
        pdf = None
        pdf_retriever = None
        pdf_chain = None
        vectorstore = None
except Exception as e:
    print(f"FAISS VectorStore 로드 중 오류 발생: {e}")
    import traceback
    traceback.print_exc()
    pdf = None
    pdf_retriever = None
    pdf_chain = None
    vectorstore = None
