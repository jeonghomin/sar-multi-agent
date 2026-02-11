"""InSAR 처리 메인 노드"""
import os
import re
from pathlib import Path

from langchain_core.messages import AIMessage
from .insar_executor import execute_insar_processing
from .insar_validation import (
    find_safe_files_in_folder,
    extract_safe_files_from_documents,
    validate_safe_files
)


def _extract_date(filename):
    m = re.search(r'(\d{8})', filename)
    return m.group(1) if m else "unknown"


def _build_products(safe_files):
    """SAFE 파일 리스트를 products 형식으로 변환"""
    return [
        {
            "index": i, "display_index": i, "original_index": i,
            "filename": f.name if isinstance(f, Path) else os.path.basename(f),
            "file_path": str(f),
            "date": _extract_date(f.name if isinstance(f, Path) else os.path.basename(f))
        }
        for i, f in enumerate(safe_files)
    ]


def _build_ready_response(safe_files):
    """insar_check로 라우팅할 공통 응답 구성"""
    files = safe_files[:2] if len(safe_files) > 2 else safe_files
    return {
        "downloaded_sar_files": [str(f) for f in files],
        "sar_search_results": {
            "success": True,
            "total": len(files),
            "products": _build_products(files)
        },
        "sar_result": {"task": "insar", "status": "ready_for_check", "message": "Master/Slave 체크 필요"}
    }


def _error_response(msg):
    """에러 응답 공통 구성"""
    return {
        "generation": msg,
        "sar_result": {"task": "insar", "status": "error", "message": msg},
        "messages": [AIMessage(content=msg)]
    }


def run_insar(state):
    """InSAR 처리 - 지진, 화산 활동 등의 지표 변형 분석"""
    print("[RUN INSAR] 시작")
    question = state.get("question", "")
    coordinates = state.get("coordinates")
    location_name = state.get("location_name")
    documents = state.get("documents", [])
    metadata = state.get("metadata")
    sar_image_path = state.get("sar_image_path")
    downloaded_sar_files = state.get("downloaded_sar_files") or []

    safe_files = []
    safe_file_patterns = [
        r'(/[^\s]+/S1[AB]_[^\s]+\.zip)',
        r'(/[^\s]+\.SAFE(?:\.zip)?)',
        r'(S1[AB]_[^\s]+\.SAFE(?:\.zip)?)',
        r'(S1[AB]_[^\s]+\.zip)',
    ]

    explicit_files = []
    for pattern in safe_file_patterns:
        matches = re.findall(pattern, question)
        if matches:
            explicit_files.extend(matches)
            if len(explicit_files) >= 2:
                break

    if explicit_files and len(explicit_files) >= 2:
        print(f"[INSAR] 질문에서 SAFE 경로 추출: {len(explicit_files)}개")
        safe_files = [Path(f) for f in explicit_files[:2]]
        missing = [f for f in safe_files if not f.exists()]
        if missing:
            return _error_response(f"❌ 지정한 파일을 찾을 수 없습니다: {', '.join(str(m) for m in missing)}")
        
        # insar_master_slave_ready가 True면 바로 실행
        if state.get("insar_master_slave_ready", False):
            insar_params = state.get("insar_parameters", {})
            return execute_insar_processing(
                safe_files, location_name, coordinates,
                subswath=insar_params.get("subswath", "IW3"),
                polarization=insar_params.get("polarization", "VV"),
                first_burst=insar_params.get("first_burst", 1),
                last_burst=insar_params.get("last_burst", 4)
            )
        return _build_ready_response(safe_files)

    elif downloaded_sar_files and len(downloaded_sar_files) >= 2:
        print(f"[INSAR] 다운로드 파일 사용: {len(downloaded_sar_files)}개")
        safe_files = [os.path.join(sar_image_path, f) for f in downloaded_sar_files] if sar_image_path else downloaded_sar_files

        if state.get("insar_master_slave_ready", False):
            insar_params = state.get("insar_parameters", {})
            return execute_insar_processing(
                [Path(f) for f in safe_files[:2]], location_name, coordinates,
                subswath=insar_params.get("subswath", "IW3"),
                polarization=insar_params.get("polarization", "VV"),
                first_burst=insar_params.get("first_burst", 1),
                last_burst=insar_params.get("last_burst", 4)
            )
        return _build_ready_response(safe_files[:2])

    if sar_image_path:
        print(f"[INSAR] 폴더 경로 사용: {sar_image_path}")
        safe_files, error = find_safe_files_in_folder(sar_image_path)
        if error:
            return error
        
        # insar_master_slave_ready가 True면 바로 실행
        if state.get("insar_master_slave_ready", False) and len(safe_files) >= 2:
            insar_params = state.get("insar_parameters", {})
            return execute_insar_processing(
                safe_files[:2], location_name, coordinates,
                subswath=insar_params.get("subswath", "IW3"),
                polarization=insar_params.get("polarization", "VV"),
                first_burst=insar_params.get("first_burst", 1),
                last_burst=insar_params.get("last_burst", 4)
            )
        return _build_ready_response(safe_files)

    folder_patterns = [
        r'([/][^\s]+?)\s*(?:폴더|디렉토리|경로)(?:로|에서|의)?',
        r'([/][^\s]+?)(?:\s+|$)(?=InSAR|insar|처리)',
    ]
    extracted_folder = None
    for pattern in folder_patterns:
        match = re.search(pattern, question)
        if match:
            extracted_folder = match.group(1)
            break

    if extracted_folder:
        print(f"[INSAR] 질문에서 폴더 추출: {extracted_folder}")
        safe_files, error = find_safe_files_in_folder(extracted_folder)
        if error:
            return error
        return _build_ready_response(safe_files)

    elif documents or metadata:
        print("[INSAR] Retrieval 데이터 사용")
        if not location_name and not coordinates:
            return _error_response("❌ InSAR 처리에 지역명/좌표가 필요합니다. 예: '튀르키예 가지안테프주 2023년 2월 6일 지진 InSAR 분석해줘'")

        safe_files, file_paths = extract_safe_files_from_documents(documents, metadata)
        safe_files, error = validate_safe_files(safe_files, file_paths)
        if error:
            return error
        return execute_insar_processing(safe_files, location_name, coordinates)

    else:
        print("⚠️ [INSAR] 데이터 소스 확인 실패")
        has_file_path = bool(re.search(r'/[^\s]+\.zip|S1[AB]_[^\s]+\.zip', question))
        if has_file_path:
            paths = re.findall(r'(/[^\s]+)', question)[:5]
            path_str = ", ".join(paths) if paths else question[:100]
            return _error_response(f"❌ 파일 경로 인식 실패. 경로/형식 확인 필요: {path_str}")

        if location_name or coordinates:
            return {
                "generation": f"🔍 **{location_name or '지정 지역'}**의 SAR 데이터를 검색하고 있습니다...",
                "sar_result": {"task": "insar", "status": "need_download", "message": "자동 다운로드 시작"},
                "auto_insar_after_download": True,
                "messages": [AIMessage(content=f"🔍 **{location_name or '지정 지역'}**의 SAR 데이터를 검색하고 있습니다...")]
            }

        return _error_response("❌ InSAR 처리 데이터 없음. 파일 경로, 폴더 경로, 또는 지역+날짜 정보를 제공해주세요.")
