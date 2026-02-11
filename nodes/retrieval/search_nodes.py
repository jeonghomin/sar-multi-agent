"""웹 검색 및 location 저장 노드"""
import json
from langchain_core.documents import Document
from core.chains import web_search_tool
from ..prompt_loader import load_prompt
from core.llm_config import llm
from location_utils import extract_locations_from_text, location_to_coordinates


def _extract_content(msg):
    """메시지에서 텍스트 content 추출 (multimodal 지원)"""
    if not hasattr(msg, 'content'):
        return ""
    content = msg.content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                return item.get('text', '')
        return str(content)
    return content


def _get_question_from_state(state):
    """state에서 question 추출 (없으면 messages에서)"""
    question = state.get("question", "")
    messages = state.get("messages", [])
    if not question and messages:
        question = _extract_content(messages[-1])
    return question


def _get_recent_context(messages, size=5, max_chars=0):
    """최근 대화 context 문자열 생성"""
    result = ""
    for msg in messages[-size:]:
        c = _extract_content(msg)
        result += (c[:max_chars] + "\n") if max_chars else (c + "\n")
    return result


def web_search(state):
    """웹 검색을 수행하고 지역명과 날짜 정보를 추출합니다."""
    print("[WEB SEARCH]")
    question = _get_question_from_state(state)
    messages = state.get("messages", [])
    summary = state.get("summary", "")

    if not question:
        return {"documents": [], "generation": "질문을 입력해주세요."}

    if len(question.strip()) < 2:
        from langchain_core.messages import AIMessage
        msg = "질문이 너무 짧습니다. 더 구체적으로 질문해주세요. 예: '2023년 한국 지진 발생 지역은?'"
        return {"documents": [], "generation": msg, "messages": [AIMessage(content=msg)]}

    question_lower = question.lower()
    data_keywords = ["데이터", "data", "가져와", "받아줘", "다운로드", "download", "가져다"]

    if any(kw in question_lower for kw in data_keywords):
        intent = "sar_get_data"
    else:
        intent_prompt = load_prompt(
            "retrieval/prompts/intent_classification.txt",
            summary=summary if summary else "(없음)",
            question=question
        )
        try:
            intent_response = llm.invoke(intent_prompt)
            response_text = intent_response.content.strip() if hasattr(intent_response, 'content') else "qa"
            lines = response_text.split('\n')
            intent = "qa"
            for line in reversed(lines[-5:]):
                line_lower = line.strip().lower()
                if "sar_insar_processing" in line_lower:
                    intent = "sar_insar_processing"
                    break
                elif "sar_get_data" in line_lower:
                    intent = "sar_get_data"
                    break
                elif "sar_search_location" in line_lower:
                    intent = "sar_search_location"
                    break
                elif line_lower == "qa" or ("qa" in line_lower and len(line_lower) < 20):
                    intent = "qa"
                    break
            valid_intents = ["qa", "sar_get_data", "sar_insar_processing", "sar_search_location"]
            if intent not in valid_intents:
                intent = "qa"
        except Exception as e:
            print(f"Intent 분류 실패: {e}")
            intent = "qa"

    print(f"Intent: {intent}")

    if intent == "sar_insar_processing":
        return {"documents": [], "location_name": None, "has_location_in_search": False}

    if intent == "sar_get_data":
        locations = extract_locations_from_text(question)
        if locations:
            return {"documents": [], "location_name": locations[0], "has_location_in_search": False}
        return {"documents": [], "location_name": None, "has_location_in_search": False}

    recent_context = _get_recent_context(messages, 5, 200)
    optimize_prompt = load_prompt(
        "retrieval/prompts/query_optimization.txt",
        summary=summary if summary else "(없음)",
        recent_context=recent_context if recent_context else "(없음)",
        question=question
    )

    try:
        optimize_response = llm.invoke(optimize_prompt)
        optimized_query = optimize_response.content.strip() if hasattr(optimize_response, 'content') else question
    except Exception as e:
        print(f"쿼리 최적화 실패: {e}")
        optimized_query = question

    search_results = web_search_tool.invoke({"query": optimized_query})
    search_results_docs = [
        Document(page_content=r['content'], metadata={'source': r['url']}) for r in search_results
    ]

    location_name = None
    date_range = None
    if search_results_docs:
        combined_text = " ".join([doc.page_content for doc in search_results_docs])
        locations = extract_locations_from_text(combined_text)
        extraction_prompt = load_prompt(
            "retrieval/prompts/location_extraction.txt",
            question=question,
            content=combined_text[:1500]
        )
        try:
            response = llm.invoke(extraction_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            for line in response_text.split('\n'):
                if '지역:' in line or 'location:' in line.lower():
                    loc = line.split(':', 1)[1].strip()
                    if loc and loc.lower() not in ["없음", "none", "no", ""]:
                        location_name = loc
                elif '날짜:' in line or 'date:' in line.lower():
                    date_text = line.split(':', 1)[1].strip()
                    if date_text and '~' in date_text:
                        parts = date_text.split('~')
                        if len(parts) == 2 and len(parts[0].strip()) >= 10 and len(parts[1].strip()) >= 10:
                            date_range = {"start_date": parts[0].strip()[:10], "end_date": parts[1].strip()[:10]}
            if not location_name and locations:
                location_name = locations[0]
        except Exception as e:
            print(f"정보 추출 오류: {e}")

    if intent == "qa":
        return {
            "documents": search_results_docs,
            "location_name": None,
            "has_location_in_search": False,
            "date_range": None,
            "coordinates": None,
            "metadata": None,
            "needs_date_search": False
        }
    return {
        "documents": search_results_docs,
        "location_name": location_name,
        "has_location_in_search": location_name is not None,
        "date_range": date_range,
        "metadata": None
    }


def _build_location_result(location_name, coordinates, date_range=None):
    """공통 location 반환 dict 생성"""
    result = {
        "location_name": location_name,
        "coordinates": coordinates,
        "awaiting_master_slave_selection": False,
        "awaiting_single_sar_selection": False,
        "sar_search_results": None
    }
    if date_range:
        result["date_range"] = date_range
    return result


def save_location(state):
    """이전 대화에서 location을 추출하고 좌표로 변환하여 저장합니다."""
    print("[SAVE LOCATION]")
    question = _get_question_from_state(state)
    messages = state.get("messages", [])
    state_location_name = state.get("location_name")
    summary = state.get("summary", "")

    reference_words = ["이지역", "이 지역", "여기", "그곳", "그 지역", "해당 지역"]
    context_size = 10 if any(ref in question for ref in reference_words) else 5
    recent_context = _get_recent_context(messages, context_size)

    context_prompt = load_prompt(
        "retrieval/prompts/context_extraction.txt",
        summary=summary if summary else "(없음)",
        recent_context=recent_context,
        question=question
    )

    extracted_location = None
    extracted_date = None
    try:
        response = llm.invoke(context_prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        for line in response_text.split('\n'):
            if '지역:' in line:
                loc = line.split(':', 1)[1].strip()
                if loc and loc.lower() not in ["없음", "none", ""]:
                    extracted_location = loc
            if '날짜:' in line:
                date = line.split(':', 1)[1].strip()
                if date and date.lower() not in ["없음", "none", ""]:
                    extracted_date = date
    except Exception as e:
        print(f"LLM 추출 실패: {e}")

    if extracted_location:
        specificity_prompt = load_prompt(
            "retrieval/prompts/specificity_check.txt",
            location_name=extracted_location
        )
        try:
            response = llm.invoke(specificity_prompt)
            specificity = response.content.strip().upper() if hasattr(response, 'content') else "SPECIFIC"
            if "NOT_SPECIFIC" in specificity:
                from langchain_core.messages import AIMessage
                msg = f"📍 지역명이 너무 포괄적입니다 ({extracted_location}). SAR 검색을 위해 더 구체적인 지역명을 알려주세요. 예: 튀르키예→튀르키예 가지안테프주, 한국→경상북도 포항시"
                return {
                    "location_name": None, "coordinates": None, "generation": msg,
                    "messages": [AIMessage(content=msg)],
                    "awaiting_master_slave_selection": False, "awaiting_single_sar_selection": False,
                    "sar_search_results": None
                }
        except Exception as e:
            print(f"구체성 판단 실패: {e}")

        coords = location_to_coordinates(extracted_location)
        if coords:
            try:
                coordinates = json.loads(coords) if isinstance(coords, str) else coords
                result = _build_location_result(extracted_location, coordinates)
                if extracted_date:
                    from datetime import datetime, timedelta
                    try:
                        target_date = datetime.strptime(extracted_date, "%Y-%m-%d")
                        result["date_range"] = {
                            "start_date": (target_date - timedelta(days=365)).strftime("%Y-%m-%d"),
                            "end_date": (target_date + timedelta(days=365)).strftime("%Y-%m-%d"),
                            "event_date": extracted_date
                        }
                    except Exception:
                        result["date_range"] = {"start_date": "2022-01-01", "end_date": "2024-12-31", "event_date": extracted_date}
                return result
            except Exception as e:
                print(f"좌표 변환 실패: {e}")

    current_locations = extract_locations_from_text(question)
    if current_locations:
        coords = location_to_coordinates(current_locations[0])
        if coords:
            try:
                coordinates = json.loads(coords) if isinstance(coords, str) else coords
                return _build_location_result(current_locations[0], coordinates)
            except Exception as e:
                print(f"좌표 변환 실패: {e}")

    if state_location_name and "," in str(state_location_name):
        if messages and len(messages) >= 2:
            rev_context = _get_recent_context(messages, 5)
            filter_prompt = f"""대화에서 사용자가 SAR 데이터를 원하는 지역을 찾아주세요.
후보: {state_location_name}
최근 대화: {rev_context}
현재 질문: {question}
선택한 지역 하나만 출력:"""
            try:
                response = llm.invoke(filter_prompt)
                filtered_location = (response.content if hasattr(response, 'content') else str(response)).strip()
                coords = location_to_coordinates(filtered_location)
                if coords:
                    coordinates = json.loads(coords) if isinstance(coords, str) else coords
                    return _build_location_result(filtered_location, coordinates)
            except Exception as e:
                print(f"LLM 필터링 실패: {e}")

    if state_location_name:
        coords = location_to_coordinates(state_location_name)
        if coords:
            try:
                coordinates = json.loads(coords) if isinstance(coords, str) else coords
                return _build_location_result(state_location_name, coordinates)
            except Exception as e:
                print(f"좌표 변환 실패: {e}")

    if messages:
        for msg in reversed(messages[-3:]):
            locations = extract_locations_from_text(_extract_content(msg))
            if locations:
                coords = location_to_coordinates(locations[0])
                if coords:
                    try:
                        coordinates = json.loads(coords) if isinstance(coords, str) else coords
                        return _build_location_result(locations[0], coordinates)
                    except Exception:
                        pass

    return {
        "location_name": None, "coordinates": None,
        "generation": "지역명을 찾을 수 없습니다. 구체적인 지역명을 말씀해주세요.",
        "awaiting_master_slave_selection": False, "awaiting_single_sar_selection": False,
        "sar_search_results": None
    }
