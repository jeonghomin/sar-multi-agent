"""LangServe 클라이언트 예제"""
from langserve import RemoteRunnable
from langchain_teddynote.messages import random_uuid

# 원격 서버에 연결 (로컬 Runnable처럼 사용)
remote_graph = RemoteRunnable("http://localhost:8000/agent_cv/")

# 설정
config = {"configurable": {"thread_id": random_uuid()}}

def test_retrieval_agent():
    """Retrieval Agent 테스트"""
    print("=== Retrieval Agent 테스트 ===")
    
    inputs = {
        "question": "군 복무 기간은 얼마인가요?",
        "messages": [],
        "documents": [],
        "generation": "",
        "summary": None,
        "intent": None,
        "image_path": None,
        "vision_result": None,
        "sar_image_path": None,
        "coordinates": None,
        "location_name": None,
        "has_location_in_search": None,
        "date_range": None,
        "needs_date_search": None,
        "awaiting_download_confirmation": None,
        "awaiting_master_slave_selection": None,
        "awaiting_single_sar_selection": None,
        "sar_search_results": None,
        "sar_result": None,
        "needs_insar": None,
        "metadata": None,
        "previous_question": None,
    }
    
    # Invoke (전체 결과 한 번에)
    result = remote_graph.invoke(inputs, config=config)
    print(f"결과: {result.get('generation', '')}\n")

def test_vision_agent():
    """Vision Agent 테스트"""
    print("=== Vision Agent 테스트 ===")
    
    inputs = {
        "question": "이 SAR 이미지에서 분류(segmentation)를 해줘",
        "image_path": "/home/mjh/Copernicus-FM/Copernicus-Bench/data/copernicusbench/dfc2020_s1s2/s1/ROIs0000_test_s1_0_p2.tif",
        "messages": [],
        "documents": [],
        "generation": "",
        "summary": None,
        "intent": None,
        "vision_result": None,
        "sar_image_path": None,
        "coordinates": None,
        "location_name": None,
        "has_location_in_search": None,
        "date_range": None,
        "needs_date_search": None,
        "awaiting_download_confirmation": None,
        "awaiting_master_slave_selection": None,
        "awaiting_single_sar_selection": None,
        "sar_search_results": None,
        "sar_result": None,
        "needs_insar": None,
        "metadata": None,
        "previous_question": None,
    }
    
    result = remote_graph.invoke(inputs, config=config)
    print(f"결과: {result.get('generation', '')}\n")

def test_streaming():
    """스트리밍 테스트"""
    print("=== 스트리밍 테스트 ===")
    
    inputs = {
        "question": "SAR 이미지 처리에 대해 간단히 설명해줘",
        "messages": [],
        "documents": [],
        "generation": "",
        "summary": None,
        "intent": None,
        "image_path": None,
        "vision_result": None,
        "sar_image_path": None,
        "coordinates": None,
        "location_name": None,
        "has_location_in_search": None,
        "date_range": None,
        "needs_date_search": None,
        "awaiting_download_confirmation": None,
        "awaiting_master_slave_selection": None,
        "awaiting_single_sar_selection": None,
        "sar_search_results": None,
        "sar_result": None,
        "needs_insar": None,
        "metadata": None,
        "previous_question": None,
    }
    
    # Stream (결과를 점진적으로 받기)
    for chunk in remote_graph.stream(inputs, config=config):
        print(f"청크: {chunk}")

def test_stream_events():
    """Stream Events 테스트 (중간 단계 포함)"""
    import asyncio
    
    print("=== Stream Events 테스트 ===")
    
    inputs = {
        "question": "군 복무 관련 정보를 검색해줘",
        "messages": [],
        "documents": [],
        "generation": "",
        "summary": None,
        "intent": None,
        "image_path": None,
        "vision_result": None,
        "sar_image_path": None,
        "coordinates": None,
        "location_name": None,
        "has_location_in_search": None,
        "date_range": None,
        "needs_date_search": None,
        "awaiting_download_confirmation": None,
        "awaiting_master_slave_selection": None,
        "awaiting_single_sar_selection": None,
        "sar_search_results": None,
        "sar_result": None,
        "needs_insar": None,
        "metadata": None,
        "previous_question": None,
    }
    
    async def run_stream():
        # Stream Events (모든 중간 단계 포함)
        async for event in remote_graph.astream_events(inputs, config=config, version="v2"):
            kind = event.get("event")
            if kind == "on_chain_start":
                print(f"시작: {event.get('name')}")
            elif kind == "on_chain_end":
                print(f"완료: {event.get('name')}")
            elif kind == "on_chat_model_stream":
                content = event.get("data", {}).get("chunk", {}).get("content")
                if content:
                    print(content, end="", flush=True)
    
    asyncio.run(run_stream())

if __name__ == "__main__":
    # 테스트 실행
    try:
        test_retrieval_agent()
        # test_vision_agent()  # 이미지 경로가 있는 경우 주석 해제
        # test_streaming()
        # test_stream_events()  # 비동기 테스트
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        print("서버가 실행 중인지 확인하세요: python server.py")
