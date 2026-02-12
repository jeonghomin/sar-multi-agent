
from langgraph.graph import END, StateGraph, START
import state
import nodes
from summarize_node import should_summarize, summarize_conversation


def create_workflow():
    """워크플로우 생성"""
    workflow = StateGraph(state.GraphState)

    # ===== Retrieval Agent 노드들 =====
    workflow.add_node("web_search", nodes.web_search)
    workflow.add_node("save_location", nodes.save_location)
    workflow.add_node("download_sar", nodes.download_sar)
    workflow.add_node("extract_coordinates", nodes.extract_coordinates)
    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("grade_document", nodes.grade_document)
    workflow.add_node("generate", nodes.generate)
    workflow.add_node("rewrite", nodes.rewrite)
    workflow.add_node("casual_generate", nodes.casual_generate)  # 일상 대화 노드

    # ===== Vision Agent 노드들 =====
    workflow.add_node("vision_router", lambda state: state)  # 패스스루 노드
    workflow.add_node("run_segmentation", nodes.run_segmentation)
    workflow.add_node("run_classification", nodes.run_classification)
    workflow.add_node("run_detection", nodes.run_detection)
    workflow.add_node("vision_generate", nodes.vision_generate)
    
    # ===== SAR Processing Agent 노드들 =====
    workflow.add_node("run_insar", nodes.run_insar)
    workflow.add_node("insar_check", nodes.check_insar_master_slave)

    # ===== 엣지 연결 =====

    # 0. START → Summarize 체크 → Router
    workflow.add_node("check_summarize", lambda state: state)  # 패스스루
    workflow.add_node("summarize", summarize_conversation)
    
    workflow.add_edge(START, "check_summarize")
    
    workflow.add_conditional_edges(
        "check_summarize",
        should_summarize,
        {
            "summarize": "summarize",
            "continue": "router",
        },
    )
    
    workflow.add_edge("summarize", "router")
    
    # 1. Router 노드 (messages → question 변환) → Main Router
    def router_prepare(state):
        """Chat 메시지를 question으로 변환 + InSAR 플래그 설정"""
        from langchain_core.messages import HumanMessage
        
        messages = state.get("messages", [])
        
        # 디버그: 현재 awaiting 플래그 상태 확인
        awaiting_confirmation = state.get("awaiting_download_confirmation", False)
        awaiting_master_slave = state.get("awaiting_master_slave_selection", False)
        awaiting_single_sar = state.get("awaiting_single_sar_selection", False)
        current_intent = state.get("intent")
        
        print(f"[Router Prepare] 현재 State - intent={current_intent}, awaiting_confirmation={awaiting_confirmation}, awaiting_master_slave={awaiting_master_slave}, awaiting_single={awaiting_single_sar}")
        
        # 항상 최신 사용자 메시지를 question으로 사용
        if messages and len(messages) > 0:
            last_message = messages[-1]
            
            # HumanMessage인 경우만 question으로 사용 (AI 메시지는 제외)
            if isinstance(last_message, HumanMessage):
                if hasattr(last_message, 'content'):
                    content = last_message.content
                    
                    # content가 list인 경우 (multimodal message)
                    if isinstance(content, list):
                        # list 안에서 text 타입 찾기
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                content = item.get('text', '')
                                break
                        else:
                            # text를 못 찾으면 전체를 str로
                            content = str(content)
                    
                    print(f"[Router Prepare] 최신 사용자 메시지를 question으로 설정: {content[:100]}")
                    
                    # InSAR 키워드 감지
                    content_lower = content.lower()
                    insar_keywords = ["insar", "간섭무늬", "interferogram", "interferometry", "ground deformation", "지표변형"]
                    needs_insar = any(keyword in content_lower for keyword in insar_keywords)
                    
                    # 폴더 경로 패턴 감지 (절대 경로 또는 상대 경로)
                    import re
                    path_pattern = r'(/[^\s]+|[A-Za-z]:\\[^\s]+|\.{1,2}/[^\s]+)'
                    path_matches = re.findall(path_pattern, content)
                    sar_image_path = None
                    
                    # InSAR + 경로가 있으면 폴더 경로로 설정
                    if needs_insar and path_matches:
                        sar_image_path = path_matches[0]  # 첫 번째 경로 사용
                        print(f"[Router Prepare] ✅ InSAR + 폴더 경로 감지: {sar_image_path}")
                    
                    if needs_insar:
                        print(f"[Router Prepare] ✅ InSAR 키워드 감지 - needs_insar=True 설정")
                    
                    # 새 질문 설정
                    # NOTE: intent와 awaiting 플래그는 유지 (previous_intent로 활용하기 위해)
                    # web_search에서 새로운 intent를 분류하면 자동으로 업데이트됨
                    return {
                        "question": content, 
                        "needs_insar": needs_insar,
                        "sar_image_path": sar_image_path
                    }
        
        return {}
    
    workflow.add_node("router", router_prepare)
    workflow.add_conditional_edges(
        "router",
        nodes.main_router,
        {
            "casual_chat": "casual_generate",  # 일상 대화 → 바로 생성
            "vision": "vision_router",
            "retrieval": "retrieval_router",
            "sar_processing": "run_insar",
        },
    )

    # 2. Vision Task Router → segmentation/classification/detection
    workflow.add_conditional_edges(
        "vision_router",
        nodes.vision_task_router,
        {
            "segmentation": "run_segmentation",
            "classification": "run_classification",
            "detection": "run_detection",
        },
    )

    # 3. Vision Task 실행 후 → vision_generate
    workflow.add_edge("run_segmentation", "vision_generate")
    workflow.add_edge("run_classification", "vision_generate")
    workflow.add_edge("run_detection", "vision_generate")

    # 4. Vision Generate → END
    workflow.add_edge("vision_generate", END)
    
    # 5. InSAR 실행 후 → 조건부 라우팅 (END or download_sar or insar_check)
    def route_after_run_insar(state):
        """
        run_insar 이후 라우팅:
        - status="need_download" → download_sar (자동 데이터 검색/다운로드)
        - status="ready_for_check" → insar_check (Master/Slave 체크)
        - 그 외 → END (실행 완료 또는 에러)
        """
        sar_result = state.get("sar_result", {})
        status = sar_result.get("status")
        
        if status == "need_download":
            print("==== [InSAR: need_download - DOWNLOAD_SAR로] ====")
            return "download_sar"
        elif status == "ready_for_check":
            print("==== [InSAR: ready_for_check - INSAR_CHECK로] ====")
            return "insar_check"
        else:
            print(f"==== [InSAR: {status} - END로] ====")
            return END
    
    workflow.add_conditional_edges(
        "run_insar",
        route_after_run_insar,
        {
            "download_sar": "download_sar",
            "insar_check": "insar_check",
            END: END,
        }
    )
    
    # 5-1. insar_check 후 → 조건부 라우팅 (run_insar or END)
    def route_after_insar_check(state):
        """
        insar_check 이후 라우팅:
        - insar_master_slave_ready=True → run_insar (실행)
        - 그 외 → END (사용자 선택 대기)
        """
        ready = state.get("insar_master_slave_ready", False)
        
        if ready:
            print("==== [INSAR_CHECK: Master/Slave 준비 완료 - RUN_INSAR로] ====")
            return "run_insar"
        else:
            print("==== [INSAR_CHECK: 사용자 선택 대기 - END로] ====")
            return END
    
    workflow.add_conditional_edges(
        "insar_check",
        route_after_insar_check,
        {
            "run_insar": "run_insar",
            END: END,
        }
    )

    # 7. Retrieval 내부 라우터 → Intent 기반 라우팅
    def retrieval_router_func(state):
        """
        Retrieval 초기 라우팅:
        - awaiting_master_slave_selection / awaiting_single_sar_selection → download_sar
        - awaiting_download_confirmation → 다운로드 확인
        - "SAR 데이터", "다운로드" 키워드 → save_location (웹 검색 스킵)
        - 그 외 → web_search (일반 검색)
        """
        from location_utils import extract_locations_from_text
        
        # 0. SAR 데이터 선택 대기 중 - LLM으로 의도 판단
        awaiting_master_slave = state.get("awaiting_master_slave_selection", False)
        awaiting_single_sar = state.get("awaiting_single_sar_selection", False)
        
        if awaiting_master_slave or awaiting_single_sar:
            question = state.get("question", "")
            print("==== [Retrieval Router] SAR 데이터 선택 대기 중 - 사용자 의도 판단] ====")
            
            # LLM으로 사용자 의도 파악
            selection_type = "Master/Slave 쌍" if awaiting_master_slave else "단일 데이터"
            intent_prompt = f"""
사용자가 SAR 데이터 리스트를 받고 {selection_type} 선택을 기다리는 중입니다.

사용자 입력: {question}

사용자의 의도를 판단하세요:

1. SELECT: 리스트에서 데이터를 선택하려는 의도
   - 예: "1번", "Master 1 Slave 5", "3번 선택", "5", "첫번째"
   
2. NEW_SEARCH: 현재 선택을 취소하고 새로운 검색을 원하는 의도
   - 예: "다시 검색", "다른 지역", "재검색", "다시", "처음부터"
   
3. CANCEL: 검색/선택을 취소하려는 의도
   - 예: "취소", "안할래", "그만", "됐어"

출력: SELECT, NEW_SEARCH, CANCEL 중 하나만 출력 (설명 없이)
"""
            
            try:
                from core.llm_config import llm
                response = llm.invoke(intent_prompt)
                user_intent = response.content.strip().upper() if hasattr(response, 'content') else "SELECT"
                
                if "NEW_SEARCH" in user_intent:
                    print("==== [LLM 판단: 새로운 검색 요청 - SAVE_LOCATION] ====")
                    # awaiting 플래그는 save_location이 리셋할 것
                    return "save_location"
                elif "CANCEL" in user_intent:
                    print("==== [LLM 판단: 검색 취소 - WEB_SEARCH] ====")
                    return "web_search"
                else:  # SELECT
                    print("==== [LLM 판단: 데이터 선택 - DOWNLOAD_SAR] ====")
                    return "download_sar"
            except Exception as e:
                print(f"⚠️ LLM 의도 판단 실패: {e}, 기본값 download_sar")
                return "download_sar"
        
        # 1. 다운로드 확인 대기 중
        awaiting_confirmation = state.get("awaiting_download_confirmation", False)
        
        if awaiting_confirmation:
            print("==== [Retrieval Router] 다운로드 확인 대기 중] ====")
            question = state.get("question", "").lower()
            
            confirmation_prompt = f"""
사용자에게 "ASF에서 데이터를 다운로드 받으시겠습니까?"라고 물어본 후 사용자가 응답했습니다.

사용자 응답: {question}

사용자가 다운로드를 원하는지 판단하세요.
- YES: 다운로드를 원함 (예: "네", "예", "응", "좋아", "해줘", "다운로드", "받아줘", "가져와" 등)
- NO: 원하지 않음 (예: "아니", "아니요", "싫어", "괜찮아", "됐어" 등)

출력: YES 또는 NO만 출력하세요.
"""
            try:
                from core.llm_config import llm
                response = llm.invoke(confirmation_prompt)
                user_intent = response.content.strip().upper() if hasattr(response, 'content') else str(response).strip().upper()
                
                if "YES" in user_intent:
                    print("==== [사용자 확인: 다운로드 원함 - DOWNLOAD_SAR] ====")
                    return "download_sar"
                else:
                    print("==== [사용자 확인: 다운로드 거부 - WEB_SEARCH] ====")
                    return "web_search"
            except Exception as e:
                print(f"LLM 오류: {e}, 기본 웹 검색")
                return "web_search"
        
        # 2. SAR 데이터 요청 키워드 감지 → 바로 save_location
        question = state.get("question", "")
        sar_keywords = ["sar", "sentinel", "위성", "satellite", "데이터", "data", "다운로드", "download", "받아", "가져"]
        
        if any(keyword in question.lower() for keyword in sar_keywords):
            # 지역명 추출 시도
            locations = extract_locations_from_text(question)
            if locations and len(locations) > 0:
                print(f"==== [Retrieval Router] SAR 데이터 요청 + 지역명({locations[0]}) 발견 - SAVE_LOCATION으로 직행] ====")
                # state에 intent와 location_name 설정
                return "save_location"
        
        # 3. 기본: web_search로 (Intent 분류는 web_search에서 수행)
        print("==== [Retrieval Router] WEB_SEARCH로 이동 (Intent 분류 예정)] ====")
        return "web_search"
    
    workflow.add_node("retrieval_router", lambda state: state)
    workflow.add_conditional_edges(
        "retrieval_router",
        retrieval_router_func,
        {
            "web_search": "web_search",
            "save_location": "save_location",
            "download_sar": "download_sar",
        },
    )

    # 8. Retrieval 워크플로우
    
    # 8-1. web_search → Intent 기반 라우팅
    def route_after_web_search(state):
        """
        web_search 이후 Intent 기반 라우팅:
        - intent="sar_insar_processing" → save_location (지역 추출 → 다운로드 → InSAR)
        - intent="sar_get_data" → save_location (지역 추출 → 데이터 가져오기)
        - intent="sar_search_location" or "qa" → generate (검색 결과 출력)
        """
        intent = state.get("intent", "qa")
        
        if intent == "sar_insar_processing":
            print("==== [Intent: sar_insar_processing - SAVE_LOCATION으로 (다운로드 먼저)] ====")
            return "save_location"
        elif intent == "sar_get_data":
            print("==== [Intent: sar_get_data - SAVE_LOCATION으로] ====")
            return "save_location"
        else:
            print(f"==== [Intent: {intent} - GENERATE로] ====")
            return "generate"
    
    workflow.add_conditional_edges(
        "web_search",
        route_after_web_search,
        {
            "save_location": "save_location",
            "generate": "generate",
            "run_insar": "run_insar",
        },
    )
    
    # 8-2. generate → END or save_location or rewrite (조건부)
    def route_after_generate(state):
        """
        generate 이후 라우팅:
        1. Q&A 모드 → END
        2. SAR 모드 + DB에서 데이터 찾음 → END (파일 경로 제공 완료)
        3. SAR 모드 + 웹 검색 결과 → END (사용자 선택 대기)
        """
        intent = state.get("intent", "qa")
        metadata = state.get("metadata")
        has_location_in_search = state.get("has_location_in_search", False)
        
        # Q&A 모드는 무조건 END
        if intent == "qa":
            print("==== [Q&A 모드 - END] ====")
            return "end"
        
        # SAR 모드
        if metadata:
            # DB 검색 성공 → END (파일 경로 제공 완료)
            print("==== [SAR 모드 - DB에서 데이터 찾음, END] ====")
            return "end"
        elif has_location_in_search:
            # web search 결과 → 사용자에게 선택지 제공, END로 가서 입력 대기
            print("==== [SAR 모드 - Web Search 결과, END로 사용자 입력 대기] ====")
            return "end"
        else:
            # 기타 → END
            print("==== [SAR 모드 - 기타, END] ====")
            return "end"
    
    workflow.add_conditional_edges(
        "generate",
        route_after_generate,
        {
            "end": END,
            "hallucination": "generate",
            "relevant": END,
            "irrelevant": "rewrite",
        },
    )
    
    # 8-3. save_location → 조건부 라우팅
    def route_after_save_location(state):
        """
        save_location 이후 라우팅:
        - location_name 있음 → extract_coordinates (정상 진행)
        - location_name 없음 → END (에러 메시지 표시 후 종료)
        """
        location_name = state.get("location_name")
        
        if location_name:
            print("==== [save_location 성공 - EXTRACT_COORDINATES] ====")
            return "extract_coordinates"
        else:
            print("==== [save_location 실패 (지역 없음) - END] ====")
            return "end"
    
    workflow.add_conditional_edges(
        "save_location",
        route_after_save_location,
        {
            "extract_coordinates": "extract_coordinates",
            "end": END,
        },
    )
    
    # 8-4. extract_coordinates → retrieve (DB 우선 검색)
    workflow.add_edge("extract_coordinates", "retrieve")
    
    # 8-5. retrieve → grade_document (문서 평가 및 다운로드 확인)
    workflow.add_edge("retrieve", "grade_document")
    
    # 8-6. grade_document 후 분기
    def route_after_grade(state):
        """grade_document 후 라우팅 (InSAR 처리 포함)"""
        documents = state.get("documents", [])
        awaiting_confirmation = state.get("awaiting_download_confirmation", False)
        needs_insar = state.get("needs_insar", False)
        
        # 다운로드 확인 대기 중 (문서 없음)
        if awaiting_confirmation:
            print("==== [문서 없음 - 사용자에게 다운로드 확인 요청, END] ====")
            return "end"
        
        # 문서가 있으면 InSAR 처리 또는 generate로
        if documents and len(documents) > 0:
            if needs_insar:
                print("==== [문서 있음 + InSAR 요청 - ROUTE TO RUN_INSAR] ====")
                return "run_insar"
            else:
                print("==== [문서 있음 - ROUTE TO GENERATE] ====")
                return "generate"
        
        # 문서 없고 확인도 안 했으면 (예외) END
        print("==== [예외 상황 - END] ====")
        return "end"
    
    workflow.add_conditional_edges(
        "grade_document",
        route_after_grade,
        {
            "generate": "generate",
            "run_insar": "run_insar",
            "end": END,
        },
    )
    
    # 8-7. download_sar 후 분기 (InSAR 처리 포함)
    def route_after_download(state):
        """
        download_sar 후 라우팅:
        - auto_insar_after_download=True → END (사용자 확인 대기, awaiting_insar_confirmation 설정됨)
        - needs_insar=True (레거시) → run_insar (바로 실행)
        - 그 외 → END (일반 다운로드 완료)
        """
        auto_insar = state.get("auto_insar_after_download", False)
        needs_insar = state.get("needs_insar", False)
        awaiting_insar = state.get("awaiting_insar_confirmation", False)
        
        if auto_insar and awaiting_insar:
            # 자동 InSAR용 다운로드 완료 → 사용자 확인 대기
            print("==== [자동 InSAR 다운로드 완료 - END (사용자 확인 대기)] ====")
            return "end"
        elif needs_insar:
            # 레거시 플로우: 바로 InSAR 실행
            print("==== [다운로드 완료 + InSAR 요청 - ROUTE TO RUN_INSAR] ====")
            return "run_insar"
        else:
            print("==== [다운로드 완료 - END] ====")
            return "end"
    
    workflow.add_conditional_edges(
        "download_sar",
        route_after_download,
        {
            "run_insar": "run_insar",
            "end": END,
        },
    )
    
    # 8-8. casual_generate → END (일상 대화 완료)
    workflow.add_edge("casual_generate", END)
    
    # 8-9. rewrite → retrieve
    workflow.add_edge("rewrite", "retrieve")

    return workflow


def create_app():
    """앱 생성 (LangGraph Studio는 자체 체크포인터 사용)"""
    workflow = create_workflow()
    app = workflow.compile()  # 체크포인터 없이 컴파일 (LangGraph Studio용)
    return app


# 일반 사용을 위한 그래프 (체크포인터 포함 가능)
workflow = create_workflow()
graph = workflow.compile()  # 로컬 실행용

# LangGraph Studio용 앱 (체크포인터 없음)
app = create_app()
