"""실행 예제"""
import config
import pdf_setup
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import invoke_graph, stream_graph, random_uuid
from langchain_teddynote.graphs import visualize_graph
from langgraph.checkpoint.memory import MemorySaver
from graph import workflow  # workflow import

# 체크포인터 추가 (로컬 실행용)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

visualize_graph(app)

config = RunnableConfig(recursion_limit=15, configurable={"thread_id": random_uuid()})

# Vision Agent 테스트
vision_inputs = {
    "question": "이 SAR 이미지에서 분류(segmentation)를 해줘",
    "image_path": "/home/mjh/Copernicus-FM/Copernicus-Bench/data/copernicusbench/dfc2020_s1s2/s1/ROIs0000_test_s1_0_p2.tif",
    "messages": [],
    "documents": [],
    "generation": "",
    "vision_result": None,
}

print("=== Vision Agent 실행 ===")
result = invoke_graph(app, vision_inputs, config)
print(f"결과: {result['generation']}")

# Retrieval Agent 테스트
retrieval_inputs = {
    "question": "군 복무 기간은 얼마인가요?",
    "messages": [],
    "documents": [],
    "generation": "",
    "image_path": None,
    "vision_result": None,
}

print("\n=== Retrieval Agent 실행 ===")
result = invoke_graph(app, retrieval_inputs, config)
print(f"결과: {result['generation']}")
