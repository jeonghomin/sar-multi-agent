from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import json
from pathlib import Path
from tqdm import tqdm

def load_documents_from_directory(directory_path):
    """
    디렉토리 하위의 모든 JSON 파일을 로드합니다.
    """
    docs = []
    json_files = []
    
    print("JSON 파일 검색 중...")
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    print(f"총 {len(json_files)}개의 JSON 파일을 찾았습니다.")
    
    print("JSON 파일 로드 중...")
    failed_count = 0
    for idx, json_file in enumerate(tqdm(json_files, desc="파일 로드"), 1):
        try:
            # JSON 파일을 직접 읽어서 처리
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                # JSON을 문자열로 변환하여 Document 생성 (중요!)
                json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
                # 메타데이터에 파일 경로 추가
                doc = Document(
                    page_content=json_str,  # 문자열로 변환된 JSON
                    metadata={"source": json_file}
                )
                docs.append(doc)
        except Exception as e:
            failed_count += 1
            if failed_count <= 10:  # 처음 10개 에러만 출력
                print(f"\n파일 로드 실패 ({json_file}): {e}")
            continue
    
    if failed_count > 10:
        print(f"\n... 외 {failed_count - 10}개 파일 로드 실패")
    
    print(f"총 {len(docs)}개의 문서를 로드했습니다. (실패: {failed_count}개)")
    return docs

def main():
    directory_path = "/mnt/nas2/BinaryData/11_AWS/umbra-sar-data/sar-data"
    
    print("=" * 60)
    print("문서 로드 시작")
    print("=" * 60)
    docs = load_documents_from_directory(directory_path)
    
    if docs:
        print("\n" + "=" * 60)
        print("텍스트 분할 시작")
        print("=" * 60)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(docs)
        print(f"텍스트 분할 완료: {len(split_docs)}개 청크 생성")
    else:
        split_docs = docs
        print("분할할 문서가 없습니다.")
        return
    
    print("\n" + "=" * 60)
    print("Embedding 모델 초기화 중...")
    print("=" * 60)
    embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")
    
    print("\n" + "=" * 60)
    print("VectorStore 생성 중...")
    print("=" * 60)
    print(f"총 {len(split_docs)}개 문서에 대한 임베딩 생성 중...")
    print("(이 작업은 시간이 오래 걸릴 수 있습니다)")
    
    # 배치 단위로 처리하여 진행 상황 표시
    vector_store = None
    batch_size = 100  # 배치 크기
    
    for i in tqdm(range(0, len(split_docs), batch_size), desc="VectorStore 생성"):
        batch_docs = split_docs[i:i+batch_size]
        if vector_store is None:
            # 첫 번째 배치로 VectorStore 초기화
            vector_store = FAISS.from_documents(batch_docs, embedding=embeddings)
        else:
            # 이후 배치는 추가
            vector_store.add_documents(batch_docs)
    
    print("VectorStore 생성 완료!")
    
    print("\n" + "=" * 60)
    print("VectorStore 저장 중...")
    print("=" * 60)
    
    save_path = "./faiss_vectorstore"
    vector_store.save_local(save_path)
    print(f"VectorStore를 {save_path}에 저장했습니다.")
    
    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)
    print(f"저장된 문서 수: {len(split_docs)}개")
    print(f"저장 경로: {save_path}")

if __name__ == "__main__":
    main()
