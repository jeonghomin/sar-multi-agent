"""좌표 추출 및 변환 노드"""
from core.llm_config import llm
from location_utils import (
    extract_locations_from_text,
    location_to_coordinates,
    get_coordinates_from_related_terms
)


def _extract_location_from_question(question, llm):
    """질문에서 지역명 추출 (헬퍼 함수)"""
    prompt = f"질문에서 지역명 추출: {question}\n지역명만 출력 (없으면 '없음'):"
    try:
        response = llm.invoke(prompt)
        location = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        if location and location.lower() not in ["없음", "none", "no", ""]:
            return location
    except:
        pass
    return None


def _extract_location_from_history(messages, llm):
    """대화 히스토리에서 지역명 추출 (헬퍼 함수)"""
    if not messages:
        return None
    
    recent = messages[-5:]
    history = "\n".join([m.content for m in recent if hasattr(m, 'content')])
    prompt = f"이전 대화에서 지역명 찾기:\n{history[:500]}\n지역명만 출력 (없으면 '없음'):"
    
    try:
        response = llm.invoke(prompt)
        location = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        if location and location.lower() not in ["없음", "none", "no", ""]:
            return location
    except:
        pass
    return None


def extract_coordinates(state):
    """
    지역명을 좌표로 변환
    1. State의 location_name 우선 사용 (save_location에서 이미 설정됨)
    2. 질문에서 추출
    3. 대화 히스토리에서 추출
    """
    print("==== [EXTRACT COORDINATES] ====")
    question = state.get("question", "")
    messages = state.get("messages", [])
    state_location = state.get("location_name")
    
    # question이 없으면 messages에서 추출
    if not question and messages and len(messages) > 0:
        last_message = messages[-1]
        if hasattr(last_message, 'content'):
            content = last_message.content
            # content가 list인 경우 (multimodal message)
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        content = item.get('text', '')
                        break
                else:
                    content = str(content)
            question = content
            print(f"[Extract Coordinates] messages에서 question 추출: {question[:100]}")
    
    location_name = None
    
    # 1. State의 location_name 우선 사용 (save_location에서 이미 정확히 추출됨)
    if state_location:
        location_name = state_location
        print(f"✅ State의 location_name 사용: {location_name}")
    
    # 2. 대명사 체크
    if not location_name:
        reference_words = ["여기", "그곳", "그 지역", "here", "there"]
        has_reference = any(ref in question.lower() for ref in reference_words)
        
        if has_reference:
            location_name = _extract_location_from_history(messages, llm)
            if location_name:
                print(f"✓ 대명사 → 히스토리에서 추출: {location_name}")
    
    # 3. 질문에서 직접 추출 (마지막 수단)
    if not location_name:
        location_name = _extract_location_from_question(question, llm)
        if not location_name:
            location_name = extract_locations_from_text(question)
            location_name = location_name[0] if location_name else None
        if location_name:
            print(f"✓ 질문에서 추출: {location_name}")
    
    if not location_name:
        print("✗ 지역명 없음")
        return {"coordinates": None}
    
    # 좌표 변환
    print(f"좌표 변환 중: {location_name}")
    coordinates = location_to_coordinates(location_name, try_variants=True, verbose=False)
    
    if not coordinates:
        # 실패 시 관련 검색어 시도
        locations = extract_locations_from_text(location_name)
        if locations:
            lat, lng, address = get_coordinates_from_related_terms(locations[:3], location_name, verbose=False)
            if lat and lng:
                coordinates = {"latitude": lat, "longitude": lng, "location": address}
    
    if coordinates:
        print(f"✓ 좌표 변환 성공: ({coordinates['latitude']}, {coordinates['longitude']})")
    else:
        print(f"✗ 좌표 변환 실패")
    
    return {"coordinates": coordinates}
