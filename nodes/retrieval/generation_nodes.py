"""응답 생성 및 평가 노드"""
from core.chains import rag_chain, query_rewriter
from evaluation.graders import hallucination_grader, answer_grader
from langchain_core.messages import AIMessage


def generate(state):
    """최종 응답 생성"""
    question = state.get("question", "")
    documents = state.get("documents", [])
    coordinates = state.get("coordinates")
    metadata = state.get("metadata")
    summary = state.get("summary", "")
    has_location_in_search = state.get("has_location_in_search", False)
    location_name = state.get("location_name")
    needs_date_search_flag = state.get("needs_date_search", False)
    intent = state.get("intent", "qa")  # 기본값 qa
    
    print(f"[GENERATE] metadata={metadata is not None}, has_location_in_search={has_location_in_search}, documents={len(documents) if isinstance(documents, list) else 'not list'}, intent={intent}")
    
    # 메타데이터가 있고 웹 검색 결과가 아닌 경우만 파일 경로 반환
    # (웹 검색 후에는 metadata가 None이어야 함)
    if metadata and not has_location_in_search:
        print("[GENERATE] ✅ 메타데이터 출력 조건 만족")
        file_path = metadata.get("source", "unknown")
        distance_km = metadata.get("distance_km")
        datetime_str = metadata.get("datetime")
        satellite = metadata.get("satellite")
        coords = metadata.get("coordinates", {})
        all_results = metadata.get("all_results", [])
        
        generation = f"""✅ DB에서 SAR 데이터를 찾았습니다!

📁 **파일 경로**:
{file_path}
"""
        
        # 날짜/시간 정보 추가
        if datetime_str:
            generation += f"\n📅 **촬영 일시**: {datetime_str}"
            if satellite:
                generation += f" ({satellite})"
        
        # 거리 정보 추가
        if distance_km is not None:
            generation += f"\n📍 **검색 좌표로부터 거리**: {distance_km:.2f} km"
        
        # 좌표 정보 추가
        if coords:
            generation += f"\n🌐 **실제 좌표**: ({coords.get('latitude', 'N/A'):.4f}, {coords.get('longitude', 'N/A'):.4f})"
        
        # 추가 결과가 있으면 표시
        if all_results and len(all_results) > 1:
            generation += f"\n\n📊 **다른 후보 데이터** ({len(all_results)-1}개):"
            for i, r in enumerate(all_results[1:4], 2):  # 2~4번째
                dt_info = f" [{r.get('datetime', 'N/A')}]" if r.get('datetime') else ""
                generation += f"\n  {i}. {r.get('distance_km', 0):.2f} km{dt_info}"
        
        print(f"[GENERATE] 메타데이터 파일 경로 반환: {file_path}")
        return {
            "generation": generation,
            "messages": [AIMessage(content=generation)]
        }
    
    # 좌표만 반환하는 경우
    if coordinates and (not documents or len(documents) == 0):
        lat = coordinates.get("latitude")
        lon = coordinates.get("longitude")
        location = coordinates.get("location", "")
        
        generation = f"""
{location}의 좌표 정보:

위도 (Latitude): {lat}
경도 (Longitude): {lon}
주소: {location}
"""
        return {
            "generation": generation,
            "messages": [AIMessage(content=generation)]
        }
    
    # 날짜 정보가 필요한 경우 (DB에 데이터 없고 날짜도 없음)
    if needs_date_search_flag and not documents:
        location = location_name or (coordinates.get("location") if coordinates else "해당 지역")
        generation = f"""
ℹ️ {location}의 SAR 데이터를 찾을 수 없습니다.

정확한 날짜 정보가 있으면 더 정확하게 찾을 수 있습니다.

**구체적으로 몇월 몇일인지 알아봐드릴까요?**

💡 "예" 또는 "알아봐줘"라고 말씀해주시면 날짜 정보를 검색해드리겠습니다.
"""
        return {
            "generation": generation,
            "messages": [AIMessage(content=generation)]
        }
    
    # RAG 생성 (웹 검색 결과에만 적용, DB 검색 결과는 이미 위에서 처리됨)
    if documents and not metadata:
        # Summary를 context에 포함
        context_with_summary = documents
        if summary:
            summary_prefix = f"[이전 대화 요약]\n{summary}\n\n[검색된 문서]\n"
            # documents가 리스트인 경우
            if isinstance(documents, list):
                context_with_summary = [summary_prefix] + documents
            else:
                context_with_summary = summary_prefix + str(documents)
        
        # 디버깅: 실제 context 크기 확인
        if isinstance(documents, list):
            total_length = sum(len(str(doc)) for doc in documents)
            print(f"[GENERATE DEBUG] documents 개수: {len(documents)}, 총 길이: {total_length}자")
            for i, doc in enumerate(documents[:2]):  # 처음 2개만 샘플 출력
                content = str(doc)[:500] if hasattr(doc, 'page_content') else str(doc)[:500]
                print(f"[GENERATE DEBUG] doc[{i}] 샘플: {content[:200]}...")
        
        generation = rag_chain.invoke({
            "question": question,
            "context": context_with_summary,
        })
        print(f"[GENERATE DEBUG] 생성된 답변 길이: {len(generation)}자")
        print(f"[GENERATE DEBUG] 생성된 답변 샘플: {generation[:500]}...")
    elif not documents and not metadata:
        generation = "검색된 문서가 없습니다."
    else:
        # metadata가 있으면 위에서 이미 처리됨, 여기까지 오면 안 됨
        generation = "정보를 처리하는 중 오류가 발생했습니다."
        print(f"[GENERATE] ⚠️ 예상치 못한 경로: metadata={metadata is not None}, documents={len(documents) if isinstance(documents, list) else type(documents)}")
    
    # SAR 모드일 때만 선택지 UI 추가 (Q&A 모드는 UI 없음!)
    if intent in ["sar_get_data", "sar_search_location"] and has_location_in_search and location_name:
        # "여러 지역" 또는 잘못된 값 체크
        if location_name.lower() in ["여러 지역", "없음", "none", "multiple regions"]:
            generation += f"""

---

💡 검색 결과에 **여러 지역**이 있습니다.

원하시는 **구체적인 지역명**을 말씀해주세요.
예: "대한민국 경상북도 영천시 데이터 가져와줘"
"""
            print(f"==== [SAR 모드 ({intent}) - 여러 지역, 구체적 지역 요청] ====")
        else:
            # 단일 지역인 경우만 UI 표시
            generation += f"""

---

📍 **{location_name}** 지역에 대한 정보를 찾았습니다.

다음 중 하나를 선택해주세요:

**1️⃣ Get Data** - 이 지역의 SAR 위성 데이터 가져오기
**2️⃣ 다른 지역 찾기** - 다른 지역을 검색하고 싶어요

💡 선택하려면 "1" 또는 "Get Data"를 입력하세요.
"""
            print(f"==== [SAR 모드 ({intent}) - 사용자 선택 UI 표시] ====")
    elif intent == "qa":
        print("==== [Q&A 모드 - UI 없음] ====")
    
    return {
        "generation": generation,
        "messages": [AIMessage(content=generation)]
    }


def grade_hallucination(state):
    """환각 및 관련성 평가"""
    documents = state["documents"]
    generation = state["generation"]
    question = state.get("question", "")
    metadata = state.get("metadata")
    
    # DB에서 찾은 경우 (metadata 있음) → hallucination 체크 불필요
    if metadata:
        print("==== [DB 결과 - Hallucination 체크 스킵, RELEVANT] ====")
        return "relevant"
    
    grade = hallucination_grader.invoke({
        "documents": documents,
        "generation": generation,
    })

    if grade.binary_score == "yes":
        print("==== [HALLUCINATION] ====")
        return "hallucination"
    elif grade.binary_score == "no":
        print("==== [NO HALLUCINATION] ====")
        score = answer_grader.invoke({
            "question": question,
            "generation": generation,
        })
        if score.binary_score == "yes":
            print("==== [RELEVANCE] ====")
            return "relevant"
        else:
            print("==== [NO RELEVANCE] ====")
            return "irrelevant"


def rewrite(state):
    """질문 재작성"""
    question = state.get("question", "")
    rewritten_question = query_rewriter.invoke(question)
    return {"question": rewritten_question}
