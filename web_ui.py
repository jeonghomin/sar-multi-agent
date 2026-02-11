"""간단한 웹 UI - Agent CV 테스트용"""
import streamlit as st
from langserve import RemoteRunnable
from langchain_teddynote.messages import random_uuid

# 페이지 설정
st.set_page_config(
    page_title="Agent CV - Web UI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agent CV - Multi-Agent System")
st.markdown("SAR 이미지 처리, 컴퓨터 비전, RAG를 위한 멀티 에이전트 시스템")

# 세션 상태 초기화
if "thread_id" not in st.session_state:
    st.session_state.thread_id = random_uuid()
if "messages" not in st.session_state:
    st.session_state.messages = []
# 마지막 state 저장 (intent, awaiting 플래그 유지용)
if "last_state" not in st.session_state:
    st.session_state.last_state = {}

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 설정")
    server_url = st.text_input(
        "서버 URL",
        value="http://localhost:8000/agent_cv/",
        help="LangServe 서버 주소"
    )
    
    st.markdown("---")
    st.header("🧪 테스트 옵션")
    
    test_mode = st.selectbox(
        "테스트 모드",
        ["RAG (문서 검색)", "Vision (이미지 분석)", "SAR Processing"]
    )
    
    if test_mode == "Vision (이미지 분석)":
        image_path = st.text_input(
            "이미지 경로",
            value="/home/mjh/Copernicus-FM/Copernicus-Bench/data/copernicusbench/dfc2020_s1s2/s1/ROIs0000_test_s1_0_p2.tif",
            help="분석할 이미지의 절대 경로"
        )
    else:
        image_path = None
    
    if st.button("🔄 새 대화 시작"):
        st.session_state.thread_id = random_uuid()
        st.session_state.messages = []
        st.rerun()
    
    st.markdown(f"**Session ID:** `{st.session_state.thread_id[:8]}...`")

# 채팅 히스토리 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("질문을 입력하세요..."):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # AI 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            try:
                # RemoteRunnable 연결
                remote_graph = RemoteRunnable(server_url)
                
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                
                # 이전 state 가져오기 (세션에서)
                current_state = st.session_state.last_state
                
                # 입력 데이터 구성 (이전 state 유지하면서 새 질문 추가)
                input_data = {
                    "question": prompt,
                    "messages": [],
                    "documents": current_state.get("documents", []),
                    "generation": "",
                    "summary": current_state.get("summary"),
                    "intent": current_state.get("intent"),  # 이전 intent 유지!
                    "image_path": image_path if test_mode == "Vision (이미지 분석)" else current_state.get("image_path"),
                    "vision_result": current_state.get("vision_result"),
                    "sar_image_path": current_state.get("sar_image_path"),
                    "downloaded_sar_files": current_state.get("downloaded_sar_files"),  # 다운로드한 파일 리스트
                    "coordinates": current_state.get("coordinates"),
                    "location_name": current_state.get("location_name"),
                    "has_location_in_search": current_state.get("has_location_in_search"),
                    "date_range": current_state.get("date_range"),
                    "needs_date_search": current_state.get("needs_date_search"),
                    "awaiting_download_confirmation": current_state.get("awaiting_download_confirmation"),  # 이전 플래그 유지!
                    "awaiting_master_slave_selection": current_state.get("awaiting_master_slave_selection"),
                    "awaiting_single_sar_selection": current_state.get("awaiting_single_sar_selection"),
                    "awaiting_insar_confirmation": current_state.get("awaiting_insar_confirmation"),  # InSAR 확인 대기
                    "awaiting_insar_parameters": current_state.get("awaiting_insar_parameters"),  # InSAR 파라미터 입력 대기
                    "sar_search_results": current_state.get("sar_search_results"),
                    "sar_result": current_state.get("sar_result"),
                    "needs_insar": current_state.get("needs_insar"),
                    "auto_insar_after_download": current_state.get("auto_insar_after_download"),
                    "insar_master_slave_ready": current_state.get("insar_master_slave_ready"),
                    "insar_parameters": current_state.get("insar_parameters"),  # InSAR 처리 파라미터
                    "metadata": current_state.get("metadata"),
                    "previous_question": current_state.get("previous_question"),
                }
                
                print(f"[UI] 이전 state 로드: intent={current_state.get('intent')}, awaiting_confirmation={current_state.get('awaiting_download_confirmation')}")
                
                # API 호출
                result = remote_graph.invoke(input_data, config=config)
                
                # 결과를 세션에 저장 (다음 요청에서 사용)
                st.session_state.last_state = result
                
                # 응답 표시
                response = result.get("generation", "응답을 생성할 수 없습니다.")
                st.markdown(response)
                
                # 히스토리에 추가
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # 디버그 정보 (확장 가능)
                with st.expander("🔍 디버그 정보"):
                    st.json({
                        "intent": result.get("intent"),
                        "documents_count": len(result.get("documents", [])),
                        "vision_result": result.get("vision_result"),
                        "sar_result": result.get("sar_result"),
                    })
                
            except Exception as e:
                error_msg = f"❌ 오류 발생: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
                st.markdown("**해결 방법:**")
                st.markdown("1. 서버가 실행 중인지 확인: `python server.py`")
                st.markdown(f"2. 서버 URL 확인: `{server_url}`")
                st.markdown("3. 방화벽 또는 포트 충돌 확인")

# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Agent CV v1.0 | Powered by LangServe & LangGraph</p>
    <p>
        <a href="http://localhost:8000/docs" target="_blank">API 문서</a> | 
        <a href="http://localhost:8000" target="_blank">서버 상태</a>
    </p>
</div>
""", unsafe_allow_html=True)
