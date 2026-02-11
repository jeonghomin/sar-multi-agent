"""GraphState 정의"""
from typing import Annotated, List, Optional, Union, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """확장된 GraphState (Vision + Retrieval + SAR Processing 통합)"""
    question: Annotated[str, "question"]
    generation: Annotated[str, "generation"]
    documents: Annotated[Union[List[str], List[Document], List[Any]], "documents"]
    
    # 대화 이력 관리 (Summary + 슬라이딩 윈도우)
    summary: Annotated[Optional[str], "summary"]  # 과거 대화 요약
    messages: Annotated[list[BaseMessage], add_messages]  # 최근 N개 대화
    
    # Intent 분류 (모드 구분)
    intent: Annotated[Optional[str], "intent"]  # "qa" | "sar_get_data" | "sar_search_location"
    
    # Vision Agent용 추가 필드
    image_path: Annotated[Optional[str], "image_path"]
    vision_result: Annotated[Optional[dict], "vision_result"]
    
    # SAR InSAR 직접 처리용 (폴더 경로)
    sar_image_path: Annotated[Optional[str], "sar_image_path"]
    downloaded_sar_files: Annotated[Optional[List[str]], "downloaded_sar_files"]  # 방금 다운로드한 SAFE 파일 리스트 (InSAR용)
    
    # 좌표 정보 (SAR 이미지 검색용)
    coordinates: Annotated[Optional[dict], "coordinates"]  # {"latitude": float, "longitude": float, "location": str}
    location_name: Annotated[Optional[str], "location_name"]  # 추출된 지역명
    has_location_in_search: Annotated[Optional[bool], "has_location_in_search"]  # web_search에서 location 발견 여부
    
    # 날짜 정보 (SAR 데이터 검색/다운로드용)
    date_range: Annotated[Optional[dict], "date_range"]  # {"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD", "event_date": "YYYY-MM-DD"}
    needs_date_search: Annotated[Optional[bool], "needs_date_search"]  # 날짜 검색이 필요한지 플래그
    awaiting_download_confirmation: Annotated[Optional[bool], "awaiting_download_confirmation"]  # ASF 다운로드 확인 대기
    
    # SAR 데이터 선택 (2단계 다운로드 플로우)
    awaiting_master_slave_selection: Annotated[Optional[bool], "awaiting_master_slave_selection"]  # Master/Slave 선택 대기 (InSAR용)
    awaiting_single_sar_selection: Annotated[Optional[bool], "awaiting_single_sar_selection"]  # 단일 SAR 데이터 선택 대기 (일반용)
    awaiting_insar_confirmation: Annotated[Optional[bool], "awaiting_insar_confirmation"]  # InSAR 처리 확인 대기 (다운로드 후)
    sar_search_results: Annotated[Optional[dict], "sar_search_results"]  # 검색 결과 리스트 (임시 저장)
    
    # SAR Processing Agent용 추가 필드
    sar_result: Annotated[Optional[dict], "sar_result"]  # SAR Processing 결과
    needs_insar: Annotated[Optional[bool], "needs_insar"]  # InSAR 처리가 필요한지 플래그
    auto_insar_after_download: Annotated[Optional[bool], "auto_insar_after_download"]  # 다운로드 후 자동으로 InSAR 처리 플래그
    insar_master_slave_ready: Annotated[Optional[bool], "insar_master_slave_ready"]  # Master/Slave 선택 완료 플래그
    insar_parameters: Annotated[Optional[dict], "insar_parameters"]  # InSAR 처리 파라미터 (subswath, polarization, burst)
    awaiting_insar_parameters: Annotated[Optional[bool], "awaiting_insar_parameters"]  # InSAR 파라미터 입력 대기 플래그
    
    # 메타데이터
    metadata: Annotated[Optional[dict], "metadata"]  # SAR 메타데이터
    previous_question: Annotated[Optional[str], "previous_question"]  # 이전 질문 (컨텍스트 유지용)
