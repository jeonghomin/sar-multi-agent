"""일상 대화 노드 (Casual Chat)"""
from core.llm_config import llm
from langchain_core.messages import AIMessage, HumanMessage


def casual_generate(state):
    """일상 대화 전용 노드 (문서 검색 없이 자연스러운 대화)"""
    print("==== [CASUAL CHAT - 일상 대화 모드] ====")
    
    question = state.get("question", "")
    messages = state.get("messages", [])
    summary = state.get("summary", "")
    
    # 대화 이력 구성 (최근 5개)
    chat_history = ""
    if messages and len(messages) > 0:
        recent_messages = messages[-5:]
        for msg in recent_messages:
            role = "사용자" if isinstance(msg, HumanMessage) else "AI"
            content = msg.content if hasattr(msg, 'content') else str(msg)
            chat_history += f"{role}: {content}\n"
    
    # 요약 정보도 포함
    context_info = ""
    if summary:
        context_info = f"\n과거 대화 요약: {summary}\n"
    
    # 프롬프트 구성
    prompt = f"""당신은 SAR 위성 이미지 처리와 재난 분석을 돕는 친근하고 전문적인 AI 어시스턴트입니다.

{context_info}
최근 대화:
{chat_history}

현재 사용자 질문: {question}

역할:
- 친근하고 자연스럽게 대화하세요
- 인사에는 간단하고 따뜻하게 응답하세요
- 감사 인사에는 겸손하게 답하세요
- 일상적인 질문에도 유용한 정보를 제공하세요
- 필요시 SAR/재난 분석 관련 정보를 자연스럽게 안내하세요

자연스럽고 도움이 되는 방식으로 응답하세요:"""
    
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        print(f"[Casual Chat] 응답 생성 완료: {content[:100]}...")
        
        return {
            "generation": content,
            "messages": [AIMessage(content=content)],
        }
        
    except Exception as e:
        print(f"❌ Casual Chat 생성 실패: {e}")
        fallback_msg = "죄송합니다. 응답 생성 중 문제가 발생했습니다. 다시 시도해주세요."
        return {
            "generation": fallback_msg,
            "messages": [AIMessage(content=fallback_msg)],
        }
