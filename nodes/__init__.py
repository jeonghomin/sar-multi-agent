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
        print("==== [InSAR 파라미터 입력 대기] ====")
        
        # 기본값 체크
        if "기본" in question_lower or "default" in question_lower:
            params = {
                "subswath": "IW3",
                "polarization": "VV",
                "first_burst": 1,
                "last_burst": 4
            }
            print(f"✅ 기본값 사용: {params}")
        else:
            # 파라미터 파싱
            params = {}
            
            # IW 추출
            iw_match = re.search(r'(IW[123])', question, re.IGNORECASE)
            params["subswath"] = iw_match.group(1).upper() if iw_match else "IW3"
            
            # Polarization 추출
            pol_match = re.search(r'\b(VV|VH|HH|HV)\b', question, re.IGNORECASE)
            params["polarization"] = pol_match.group(1).upper() if pol_match else "VV"
            
            # Burst 추출
            burst_match = re.search(r'burst\s*(\d+)\s*-\s*(\d+)', question, re.IGNORECASE)
            if burst_match:
                params["first_burst"] = int(burst_match.group(1))
                params["last_burst"] = int(burst_match.group(2))
            else:
                # 단일 숫자 2개 찾기
                nums = re.findall(r'\b(\d+)\b', question)
                if len(nums) >= 2:
                    params["first_burst"] = int(nums[0])
                    params["last_burst"] = int(nums[1])
                else:
                    params["first_burst"] = 1
                    params["last_burst"] = 4
            
            print(f"✅ 파싱된 파라미터: {params}")
        
        # state 업데이트하고 run_insar로 라우팅
        state["insar_parameters"] = params
        state["awaiting_insar_parameters"] = False
        state["insar_master_slave_ready"] = True
        
        print("==== [InSAR 파라미터 설정 완료 - SAR PROCESSING로] ====")
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
    
    # 우선순위 1: InSAR 키워드 또는 파일 경로 빠른 체크 (⭐ 최우선!)
    insar_keywords = ["insar", "간섭무늬", "interferogram", "지표변형", "master", "slave"]
    has_insar_keyword = any(keyword in question_lower for keyword in insar_keywords)
    
    # Sentinel-1 파일 경로 패턴 체크
    has_sentinel_file = bool(re.search(r'S1[AB]_[^\s]+\.(?:zip|SAFE)', question))
    
    # 파일 경로가 명시되어 있으면 바로 SAR PROCESSING으로
    if has_sentinel_file:
        print(f"==== [Sentinel 파일 경로 감지 - SAR PROCESSING로 직행] ====")
        return "sar_processing"
    
    # InSAR 키워드만 있고 파일 경로가 없으면 RETRIEVAL로 (웹 검색 & 다운로드)
    if has_insar_keyword:
        print(f"==== [InSAR 키워드 감지 - RETRIEVAL로 (검색 & 다운로드)] ====")
        return "retrieval"
    
    # 우선순위 2: 질문 유형 판단 (Q&A 질문인지 확인)
    # - "어디", "뭐", "어떤", "언제" 같은 의문사 → Q&A 질문
    qa_keywords = ["어디", "어떤", "뭐", "무엇", "언제", "왜", "어떻게", "얼마",
                   "where", "what", "when", "why", "how", "which"]
    is_qa_question = any(keyword in question_lower for keyword in qa_keywords)
    
    if is_qa_question:
        print(f"==== [Q&A 의문사 감지 - LLM 라우터로 새로운 Intent 판단] ====")
        # LLM 라우터로 넘어가서 정확한 intent 판단
    
    # 우선순위 3: Awaiting 플래그 기반 처리
    # - Q&A 질문이 아니고 + 선택 대기 상태이면 retrieval로 진행
    elif is_awaiting:
        print(f"==== [Awaiting 플래그 감지 - RETRIEVAL로 계속 진행] ====")
        return "retrieval"
    
    # 우선순위 4: Previous Intent + 데이터 요청 키워드 기반 라우팅
    # - 이전에 SAR 데이터 요청이 있었고 + 데이터 요청 키워드가 있으면 → retrieval로 진행
    elif previous_intent in ["sar_get_data", "sar_search_location"]:
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
    
    # 우선순위 5 (최종): 모든 경우 → LLM 라우터 사용 (Intent 판단)
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
