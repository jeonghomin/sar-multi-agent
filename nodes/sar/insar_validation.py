"""InSAR SAFE 파일 검증 로직"""
import re
from pathlib import Path
from langchain_core.messages import AIMessage


def find_safe_files_in_folder(folder_path: str):
    """
    폴더에서 SAFE 파일 찾기
    
    Args:
        folder_path: 폴더 경로
    
    Returns:
        tuple: (safe_files: list, error_dict: dict or None)
    """
    folder = Path(folder_path)
    
    if not folder.exists() or not folder.is_dir():
        error_msg = f"❌ 폴더 경로가 유효하지 않습니다: {folder_path}"
        print(error_msg)
        return None, {
            "generation": error_msg,  # web_ui.py에서 표시용
            "sar_result": {
                "task": "insar",
                "status": "error",
                "message": error_msg
            },
            "messages": [AIMessage(content=error_msg)]
        }
    
    # 폴더에서 .SAFE.zip 파일 찾기
    safe_files = list(folder.glob("*.SAFE.zip")) + list(folder.glob("*.SAFE"))
    
    if len(safe_files) != 2:
        error_msg = f"""❌ InSAR 처리를 위해서는 **정확히 2개의 SAFE.zip 파일**이 필요합니다.

📁 현재 폴더: `{folder_path}`
📊 찾은 파일 개수: {len(safe_files)}개

**요구사항:**
- Sentinel-1 SLC 데이터 형식: `.SAFE.zip` 또는 `.SAFE`
- **정확히 2개**의 SAR 이미지 (간섭무늬 생성용)

**해결 방법:**
1. 정확히 2개의 SAFE.zip 파일을 준비해주세요
2. ASF Data Search에서 Sentinel-1 SLC 데이터를 다운로드하세요
3. 또는 **구체적인 지역+날짜**를 말씀하시면 자동으로 검색/다운로드합니다
   - 예: "튀르키예 가지안테프주 2023년 2월 6일 지진 InSAR 분석해줘"

현재 폴더의 SAFE 파일:
{chr(10).join([f"- {f.name}" for f in safe_files]) if safe_files else "(없음)"}
"""
        print(error_msg)
        return None, {
            "generation": error_msg,  # web_ui.py에서 표시용
            "sar_result": {
                "task": "insar",
                "status": "error",
                "message": error_msg
            },
            "messages": [AIMessage(content=error_msg)]
        }
    
    print(f"✅ 2개의 SAFE 파일 발견:")
    for f in safe_files:
        print(f"  - {f.name}")
    
    return safe_files, None


def extract_safe_files_from_documents(documents, metadata):
    """
    documents와 metadata에서 SAFE 파일 경로 추출
    
    Args:
        documents: 문서 리스트
        metadata: 메타데이터 딕셔너리
    
    Returns:
        list: SAFE 파일 경로 리스트 (Path 객체)
    """
    file_paths = []
    
    # metadata에서 추출
    if metadata:
        file_path = metadata.get("file_path", "")
        if file_path:
            file_paths.append(Path(file_path))
    
    # documents에서 추출
    if documents and len(documents) > 0:
        for doc in documents[:10]:  # 최대 10개까지만 확인
            doc_str = str(doc)
            # 경로 패턴 추출 시도
            paths = re.findall(r'(/[^\s]+\.SAFE(?:\.zip)?|[A-Za-z]:\\[^\s]+\.SAFE(?:\.zip)?)', doc_str)
            for p in paths:
                file_paths.append(Path(p))
    
    print(f"📂 추출된 파일 경로: {len(file_paths)}개")
    for fp in file_paths:
        print(f"  - {fp}")
    
    # SAFE 포맷 파일만 필터링
    safe_files = [fp for fp in file_paths if fp.suffix in ['.zip', '.SAFE'] and 'SAFE' in fp.name]
    
    print(f"✅ SAFE 포맷 파일: {len(safe_files)}개")
    
    return safe_files, file_paths


def validate_safe_files(safe_files, all_file_paths):
    """
    SAFE 파일 개수 및 존재 여부 검증
    
    Args:
        safe_files: SAFE 파일 리스트
        all_file_paths: 전체 파일 경로 리스트
    
    Returns:
        tuple: (validated_safe_files: list or None, error_dict: dict or None)
    """
    # SAFE 파일이 없으면 에러
    if len(safe_files) == 0:
        error_msg = f"""❌ InSAR 처리를 위해서는 **Sentinel-1 SAFE 포맷** 데이터가 필요합니다.

📊 DB/다운로드에서 찾은 파일: {len(all_file_paths)}개
❌ SAFE 포맷 파일: 0개

**InSAR 처리 요구사항:**
- Sentinel-1 SLC 데이터 형식: `.SAFE.zip` 또는 `.SAFE`
- **정확히 2개**의 SAR 이미지 (간섭무늬 생성용)
- **정확한 지역명 + 날짜** 정보 필요

**해결 방법:**
1. ASF Data Search에서 Sentinel-1 **SLC** 데이터를 다운로드하세요
   - GRD 형식이 아닌 **SLC (Single Look Complex)** 형식 필요
2. 또는 **구체적인 지역+날짜**로 요청하면 자동으로 검색/다운로드합니다
   - 예: "튀르키예 가지안테프주 2023년 2월 6일 지진 InSAR 분석해줘"
3. 또는 SAFE 포맷 파일이 있는 폴더 경로를 직접 제공해주세요

현재 찾은 파일:
{chr(10).join([f"- {fp}" for fp in all_file_paths]) if all_file_paths else "(없음)"}
"""
        print(error_msg)
        return None, {
            "generation": error_msg,  # web_ui.py에서 표시용
            "sar_result": {
                "task": "insar",
                "status": "error",
                "message": error_msg
            },
            "messages": [AIMessage(content=error_msg)]
        }
    
    # SAFE 파일이 2개가 아니면 경고
    if len(safe_files) < 2:
        error_msg = f"""⚠️ InSAR 처리를 위해서는 **정확히 2개의 SAFE 파일**이 필요합니다.

📊 현재 찾은 SAFE 파일: {len(safe_files)}개

**InSAR 처리 요구사항:**
- **정확히 2개**의 SAR 이미지 필요 (간섭무늬 생성용)
- 동일 지역, 다른 시간대 촬영본

**해결 방법:**
1. 동일 지역의 다른 날짜 SAR 데이터를 추가로 다운로드하세요
2. 또는 **구체적인 지역+날짜**로 다시 요청하면 자동으로 2개를 찾습니다
   - 예: "튀르키예 가지안테프주 2023년 2월 6일 지진 InSAR 분석해줘"
3. 또는 2개의 SAFE 파일이 있는 폴더 경로를 직접 제공해주세요

현재 SAFE 파일:
{chr(10).join([f"- {sf.name}" for sf in safe_files])}
"""
        print(error_msg)
        return None, {
            "generation": error_msg,  # web_ui.py에서 표시용
            "sar_result": {
                "task": "insar",
                "status": "error",
                "message": error_msg
            },
            "messages": [AIMessage(content=error_msg)]
        }
    
    # 2개 이상이면 처음 2개만 사용
    if len(safe_files) > 2:
        print(f"⚠️ SAFE 파일이 {len(safe_files)}개 발견됨. 처음 2개만 사용합니다.")
        safe_files = safe_files[:2]
    
    print(f"✅ InSAR 처리에 사용할 SAFE 파일 2개:")
    for f in safe_files:
        print(f"  - {f.name}")
    
    # 파일이 실제로 존재하는지 확인
    for sf in safe_files:
        if not sf.exists():
            error_msg = f"❌ 파일을 찾을 수 없습니다: {sf}"
            print(error_msg)
            return None, {
                "generation": error_msg,  # web_ui.py에서 표시용
                "sar_result": {
                    "task": "insar",
                    "status": "error",
                    "message": error_msg
                },
                "messages": [AIMessage(content=error_msg)]
            }
    
    return safe_files, None
