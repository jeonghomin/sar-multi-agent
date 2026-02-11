"""
대화 이력 요약 노드 (Summary + 슬라이딩 윈도우)
"""
from core.llm_config import llm_summary


# 설정값
MAX_MESSAGES = 10  # 최근 메시지 유지 개수
SUMMARIZE_THRESHOLD = 8  # 요약 시작 임계값


def should_summarize(state) -> str:
    """
    대화 이력이 일정 길이 이상이면 요약 필요 여부 판단
    
    Returns:
        "summarize" or "continue"
    """
    messages = state.get("messages", [])
    
    if len(messages) > SUMMARIZE_THRESHOLD:
        print(f"==== [대화 길이 {len(messages)}개 - 요약 필요] ====")
        return "summarize"
    else:
        print(f"==== [대화 길이 {len(messages)}개 - 요약 불필요] ====")
        return "continue"


def summarize_conversation(state):
    """
    대화 이력 요약 (Summary + 슬라이딩 윈도우)
    
    1. 기존 summary + 오래된 messages를 요약
    2. summary 업데이트
    3. messages는 최근 MAX_MESSAGES개만 유지
    """
    print("==== [SUMMARIZE CONVERSATION] ====")
    
    messages = state.get("messages", [])
    existing_summary = state.get("summary", "")
    
    if len(messages) <= MAX_MESSAGES:
        # 요약 불필요
        print("메시지가 충분히 적음 - 요약 스킵")
        return state
    
    # 최근 메시지와 오래된 메시지 분리
    recent_messages = messages[-MAX_MESSAGES:]
    old_messages = messages[:-MAX_MESSAGES]
    
    # 요약할 내용 구성
    old_messages_text = "\n".join([
        f"{msg.type}: {msg.content}" 
        for msg in old_messages 
        if hasattr(msg, 'content')
    ])
    
    # 요약 프롬프트
    summarize_prompt = f"""
다음은 이전 대화 내용입니다. 중요한 정보를 유지하면서 간결하게 요약해주세요.

기존 요약:
{existing_summary if existing_summary else "없음"}

추가 대화 내용:
{old_messages_text}

요약 시 포함할 내용:
1. 사용자가 관심있어 하는 주제/지역
2. 이미 제공된 정보 (SAR 데이터, 좌표 등)
3. 진행 중인 작업 상태
4. 중요한 컨텍스트

간결하게 3-5문장으로 요약하세요.
"""
    
    try:
        # LLM으로 요약 생성 (temperature=0 for consistency)
        summary_response = llm_summary.invoke(summarize_prompt)
        new_summary = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)
        
        print(f"요약 완료: {len(messages)}개 → 최근 {MAX_MESSAGES}개 유지")
        print(f"새 요약: {new_summary[:100]}...")
        
        return {
            "summary": new_summary,
            "messages": recent_messages,
        }
        
    except Exception as e:
        print(f"요약 중 오류: {e}")
        # 오류 시 그냥 최근 메시지만 유지
        return {
            "messages": recent_messages,
        }


def get_conversation_context(state) -> str:
    """
    대화 컨텍스트 생성 (프롬프트에 사용)
    
    Returns:
        "[대화 요약]\n{summary}\n\n[최근 대화]\n{messages}"
    """
    summary = state.get("summary", "")
    messages = state.get("messages", [])
    
    context = ""
    
    # 요약 추가
    if summary:
        context += f"[대화 요약]\n{summary}\n\n"
    
    # 최근 대화 추가
    if messages:
        context += "[최근 대화]\n"
        for msg in messages[-5:]:  # 최근 5개만
            if hasattr(msg, 'content'):
                role = "사용자" if msg.type == "human" else "AI"
                content = msg.content[:200]  # 길이 제한
                context += f"{role}: {content}\n"
    
    return context if context else "대화 시작"


def format_messages_for_prompt(state) -> list:
    """
    프롬프트용 메시지 포맷팅 (summary 포함)
    
    LLM에 전달할 때 summary를 첫 메시지로 추가
    """
    from langchain_core.messages import SystemMessage
    
    summary = state.get("summary", "")
    messages = state.get("messages", [])
    
    if not summary:
        return messages
    
    # Summary를 시스템 메시지로 추가
    summary_message = SystemMessage(content=f"""
[이전 대화 요약]
{summary}

위 요약을 참고하여 대화를 계속하세요.
""")
    
    return [summary_message] + messages
