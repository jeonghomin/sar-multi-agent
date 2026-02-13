"""노드 모듈 - 모든 노드 함수"""
import re

from .retrieval import (
    web_search,
    save_location,
    extract_coordinates,
    retrieve,
    grade_document,
    generate,
    grade_hallucination,
    rewrite,
    download_sar
)

from .retrieval.casual_chat_node import casual_generate

from .vision.vision_nodes import (
    vision_task_router,
    run_segmentation,
    run_classification,
    run_detection,
    vision_generate
)

from .sar import (
    run_insar,
    check_insar_master_slave,
)


# ===== Main Router =====
def main_router(state):
    """최상위 라우터: Vision Agent vs Retrieval Agent vs SAR Processing Agent 선택"""
    print("==== [MAIN ROUTER] ====")
    question = state.get("question", "")
    sar_image_path = state.get("sar_image_path")
    
    # question이 list인 경우 처리
    if isinstance(question, list):
        for item in question:
            if isinstance(item, dict) and item.get('type') == 'text':
                question = item.get('text', '')
                break
        else:
            question = str(question)
    
    if not question or not isinstance(question, str):
        print("==== [WARNING: No valid question provided, defaulting to retrieval] ====")
        return "retrieval"
    
    previous_question = state.get("previous_question", "")
    question_lower = question.lower()
    previous_intent = state.get("intent")  # 이전 intent 가져오기
    
    # awaiting 플래그 체크
    awaiting_confirmation = state.get("awaiting_download_confirmation", False)
    awaiting_master_slave = state.get("awaiting_master_slave_selection", False)
    awaiting_single_sar = state.get("awaiting_single_sar_selection", False)
    awaiting_insar = state.get("awaiting_insar_confirmation", False)
    awaiting_insar_params = state.get("awaiting_insar_parameters", False)
    is_awaiting = awaiting_confirmation or awaiting_master_slave or awaiting_single_sar or awaiting_insar or awaiting_insar_params
    
    print(f"[DEBUG] 질문: {question}")
    print(f"[DEBUG] sar_image_path: {sar_image_path}")
    print(f"[DEBUG] previous_intent: {previous_intent}")
    print(f"[DEBUG] is_awaiting: {is_awaiting} (confirmation={awaiting_confirmation}, master_slave={awaiting_master_slave}, single={awaiting_single_sar}, insar={awaiting_insar}, insar_params={awaiting_insar_params})")
    
    # 우선순위 0-1: InSAR 파라미터 입력 대기
    if awaiting_insar_params:
        print("==== [InSAR 파라미터 입력 대기 - SAR PROCESSING로] ====")
        # ⭐ insar_check에서 파라미터 파싱 처리
        return "sar_processing"
    
    # 우선순위 0-2: InSAR 확인 대기 - 긍정적 응답 시 InSAR 처리
    if awaiting_insar:
        # 긍정적인 응답 체크
        positive_responses = ["네", "yes", "y", "ok", "진행", "좋아", "응", "ㅇ", "ㅇㅋ", "ㅇㅇ", "예", "확인"]
        is_positive = any(resp in question_lower for resp in positive_responses)
        
        if is_positive:
            print("==== [InSAR 확인 대기 - 긍정 응답 → SAR PROCESSING로] ====")
            return "sar_processing"
        else:
            # 취소 또는 다른 질문
            print("==== [InSAR 확인 대기 - 취소 또는 다른 질문 → RETRIEVAL로] ====")
            return "retrieval"
    
    # 우선순위 1: Awaiting 플래그 기반 처리
    # - 선택 대기 상태이면 retrieval로 진행
    if is_awaiting:
        print(f"==== [Awaiting 플래그 감지 - RETRIEVAL로 계속 진행] ====")
        return "retrieval"
    
    # 우선순위 2: Previous Intent + 데이터 요청 키워드 기반 라우팅
    # - 이전에 SAR 데이터 요청이 있었고 + 데이터 요청 키워드가 있으면 → retrieval로 진행
    if previous_intent in ["sar_get_data", "sar_search_location"]:
        # 데이터 요청 관련 키워드 체크
        data_request_keywords = ["데이터", "data", "가져와", "받아", "다운로드", "download", 
                                 "주", "시", "도", "현", "구", "군", "읍", "면", "동",  # 지역명
                                 "년", "월", "일", "date"]  # 날짜
        
        has_data_request = any(keyword in question_lower for keyword in data_request_keywords)
        
        if has_data_request:
            print(f"==== [Previous Intent={previous_intent} + 데이터 요청 키워드 - RETRIEVAL로 계속 진행] ====")
            return "retrieval"
        else:
            print(f"==== [Previous Intent={previous_intent} BUT 데이터 요청 키워드 없음 - 새로운 Intent 판단] ====")
    
    # 우선순위 3 (최종): 모든 경우 → LLM 라우터 사용 (Intent 판단)
    try:
        from routing import main_agent as router_agent
        messages = state.get("messages", [])
        summary = state.get("summary", "")  # ✅ Summary 가져오기
        source = router_agent.invoke({
            "question": question,
            "messages": messages,
            "summary": summary  # ✅ Summary 전달
        })
        
        # source가 None이거나 datasource 속성이 없으면 기본값
        if source is None:
            print("==== [WARNING: LLM router returned None, defaulting to retrieval] ====")
            return "retrieval"
        
        if not hasattr(source, 'datasource'):
            print(f"==== [WARNING: LLM router response has no datasource attribute: {source}, defaulting to retrieval] ====")
            return "retrieval"
        
        datasource = source.datasource
        
        if datasource == "casual_chat":
            print("==== [LLM: ROUTE TO CASUAL CHAT] ====")
            return "casual_chat"
        elif datasource == "vision":
            print("==== [LLM: ROUTE TO VISION AGENT] ====")
            return "vision"
        elif datasource == "sar_processing":
            print("==== [LLM: ROUTE TO SAR PROCESSING AGENT] ====")
            return "sar_processing"
        else:  # retrieval
            print("==== [LLM: ROUTE TO RETRIEVAL AGENT] ====")
            return "retrieval"
    except Exception as e:
        print(f"==== [ERROR in main_router: {e}, defaulting to retrieval] ====")
        return "retrieval"


__all__ = [
    # Main Router
    'main_router',
    
    # Retrieval Nodes
    'web_search',
    'save_location',
    'download_sar',
    'extract_coordinates',
    'retrieve',
    'grade_document',
    'generate',
    'grade_hallucination',
    'rewrite',
    'casual_generate',  # 일상 대화 노드
    
    # Vision Nodes
    'vision_task_router',
    'run_segmentation',
    'run_classification',
    'run_detection',
    'vision_generate',
    
    # SAR Nodes
    'run_insar',
    'check_insar_master_slave',
]
