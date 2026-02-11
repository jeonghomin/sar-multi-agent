"""Vision Agent 노드 함수들"""
from routing.routers import vision_router
from core.llm_config import llm_vision
from sar_segmentation_node import run_sar_segmentation, sar_segmentation_generate
import vision_tools


def vision_task_router(state):
    """Vision 작업 라우팅: segmentation, classification, detection 중 선택"""
    print("==== [VISION TASK ROUTER] ====")
    question = state.get("question", "")
    
    # 명시적 키워드 체크 (우선순위)
    if any(kw in question.lower() for kw in ["segmentation", "분할", "lulc", "land cover"]):
        print("==== [Segmentation 키워드 감지 - SELECTED TASK: segmentation] ====")
        return "segmentation"
    elif any(kw in question.lower() for kw in ["classification", "분류", "classify"]):
        print("==== [Classification 키워드 감지 - SELECTED TASK: classification] ====")
        return "classification"
    elif any(kw in question.lower() for kw in ["detection", "탐지", "detect", "찾기", "finding"]):
        print("==== [Detection 키워드 감지 - SELECTED TASK: detection] ====")
        return "detection"
    
    try:
        messages = state.get("messages", [])
        summary = state.get("summary", "")  # ✅ Summary 가져오기
        result = vision_router.invoke({
            "question": question,
            "messages": messages,
            "summary": summary  # ✅ Summary 전달
        })
        task = result.task
        print(f"==== [SELECTED TASK: {task}] ====")
        
        if task == "segmentation":
            return "segmentation"
        elif task == "classification":
            return "classification"
        else:  # detection
            return "detection"
    except Exception as e:
        print(f"==== [ERROR in vision_task_router: {e}, defaulting to segmentation] ====")
        return "segmentation"


def run_segmentation(state):
    """Segmentation Tool 직접 호출 (SAR 이미지는 ai-service API 사용)"""
    print("==== [RUN SEGMENTATION] ====")
    image_path = state.get("image_path", "")
    question = state.get("question", "")
    use_gt = state.get("use_gt", True)  # 기본적으로 GT 모드 사용
    
    print(f"[DEBUG] state에서 받은 image_path: {image_path}")
    print(f"[DEBUG] image_path 타입: {type(image_path)}")
    print(f"[DEBUG] question: {question}")
    
    if not image_path:
        error_msg = "이미지 경로가 state.image_path에 설정되지 않았습니다. LangGraph Studio에서 image_path 입력 필드에 이미지 경로를 입력해주세요."
        return {"vision_result": {"error": error_msg, "status": "no_image_path"}}
    
    # SAR 이미지 감지
    image_path_lower = image_path.lower()
    is_sar_image = (
        "_s1_" in image_path_lower or 
        "sentinel-1" in image_path_lower or 
        "/s1/" in image_path_lower or
        "sar" in image_path_lower
    )
    
    if is_sar_image:
        print("==== [SAR 이미지 감지 - ai-service API 사용] ====")
        return run_sar_segmentation(state)
    
    # 일반 이미지는 기존 vision_tools 사용
    try:
        print(f"[INFO] Segmentation 처리 시작: {image_path}")
        result = vision_tools.segmentation_tool.invoke(image_path)
        print(f"[INFO] Segmentation 결과: {result}")
        
        # tool에서 반환한 에러 확인
        if isinstance(result, dict) and "error" in result:
            print(f"[WARNING] Tool에서 에러 반환: {result}")
            return {"vision_result": result}
        
        if not result or (isinstance(result, dict) and not result.get("output_path")):
            print(f"[WARNING] Segmentation 결과가 비어있거나 output_path가 없음")
            return {"vision_result": {"error": "Segmentation 처리는 완료되었으나 결과가 비어있습니다.", "status": "empty_result"}}
        
        return {"vision_result": result}
    except Exception as e:
        print(f"[ERROR] Segmentation 에러: {e}")
        import traceback
        traceback.print_exc()
        return {"vision_result": {"error": f"Segmentation 처리 중 오류: {str(e)}", "status": "failed"}}


def run_classification(state):
    """Classification Tool 직접 호출"""
    print("==== [RUN CLASSIFICATION] ====")
    image_path = state.get("image_path", "")
    
    if not image_path:
        error_msg = "이미지 경로가 state.image_path에 설정되지 않았습니다. LangGraph Studio에서 image_path 입력 필드에 이미지 경로를 입력해주세요."
        return {"vision_result": {"error": error_msg, "status": "no_image_path"}}
    
    try:
        print(f"[INFO] Classification 처리 시작: {image_path}")
        result = vision_tools.classification_tool.invoke(image_path)
        print(f"[INFO] Classification 결과: {result}")
        
        # tool에서 반환한 에러 확인
        if isinstance(result, dict) and "error" in result:
            print(f"[WARNING] Tool에서 에러 반환: {result}")
            return {"vision_result": result}
        
        return {"vision_result": result}
    except Exception as e:
        print(f"[ERROR] Classification 에러: {e}")
        import traceback
        traceback.print_exc()
        return {"vision_result": {"error": f"Classification 처리 중 오류: {str(e)}", "status": "failed"}}


def run_detection(state):
    """Detection Tool 직접 호출"""
    print("==== [RUN DETECTION] ====")
    image_path = state.get("image_path", "")
    
    if not image_path:
        error_msg = "이미지 경로가 state.image_path에 설정되지 않았습니다. LangGraph Studio에서 image_path 입력 필드에 이미지 경로를 입력해주세요."
        return {"vision_result": {"error": error_msg, "status": "no_image_path"}}
    
    try:
        print(f"[INFO] Detection 처리 시작: {image_path}")
        result = vision_tools.detection_tool.invoke(image_path)
        print(f"[INFO] Detection 결과: {result}")
        
        # tool에서 반환한 에러 확인
        if isinstance(result, dict) and "error" in result:
            print(f"[WARNING] Tool에서 에러 반환: {result}")
            return {"vision_result": result}
        
        return {"vision_result": result}
    except Exception as e:
        print(f"[ERROR] Detection 에러: {e}")
        import traceback
        traceback.print_exc()
        return {"vision_result": {"error": f"Detection 처리 중 오류: {str(e)}", "status": "failed"}}


def vision_generate(state):
    """Vision Tool 결과를 바탕으로 최종 응답 생성"""
    print("==== [VISION GENERATE] ====")
    question = state.get("question", "")
    vision_result = state.get("vision_result", {})
    messages = state.get("messages", [])
    summary = state.get("summary", "")  # ✅ Summary 가져오기
    
    # 에러 체크
    if "error" in vision_result:
        error_msg = vision_result.get("error", "알 수 없는 오류")
        status = vision_result.get("status", "failed")
        
        # 이미지 경로가 설정되지 않은 경우
        if status == "no_image_path":
            generation = f"""
❌ 이미지 경로가 설정되지 않았습니다.

**LangGraph Studio 사용 방법:**
1. 입력 창 상단에 `image_path` 필드가 있습니다
2. 해당 필드에 이미지 파일 경로를 입력해주세요
   예: `/home/mjh/Project/LLM/RAG/files/test_folder/ROIs0000_test_s1_0_p1004.tif`
3. `question` 필드에는 원하는 작업을 입력해주세요
   예: "SAR 이미지 분할 해줘"

**입력 예시:**
```json
{{
  "question": "SAR 이미지 분할 해줘",
  "image_path": "/home/mjh/Project/LLM/RAG/files/test_folder/ROIs0000_test_s1_0_p1004.tif"
}}
```
"""
        elif status == "connection_error":
            server_url = vision_result.get("server_url", "Unknown")
            generation = f"""
❌ FastAPI 서버에 연결할 수 없습니다.

**서버 주소:** {server_url}

**해결 방법:**
1. FastAPI 서버가 실행 중인지 확인해주세요
   ```bash
   # 서버 실행 명령 예시
   cd /home/mjh/Project/LLM/RAG/ai-service
   uvicorn app.main:app --host 192.168.10.174 --port 6600
   ```

2. 서버 주소와 포트가 올바른지 확인해주세요
   - 현재 설정: {server_url}
   - vision_tools.py의 FASTAPI_BASE_URL 확인

3. 방화벽이나 네트워크 설정을 확인해주세요

4. 서버 상태 확인:
   ```bash
   curl {server_url}/health
   ```
"""
        else:
            generation = f"""
❌ 이미지 처리 중 오류가 발생했습니다.

**오류 내용:** {error_msg}

**해결 방법:**
1. 이미지 경로가 올바른지 확인해주세요
2. 이미지 파일이 존재하는지 확인해주세요
3. 파일 형식이 지원되는지 확인해주세요 (PNG, JPG, TIFF)
4. 파일에 읽기 권한이 있는지 확인해주세요
"""
        return {"generation": generation, "previous_question": question}
    
    # SAR Segmentation 결과 감지 (lulc_summary 필드가 있으면 SAR 결과)
    if "lulc_summary" in vision_result:
        return sar_segmentation_generate(state)
    
    # 이미지 경로 추출 (LangGraph Studio에서 확인 가능하도록)
    output_path = vision_result.get("output_path", "")
    analysis_result = vision_result.get("analysis_result", {})
    
    # 결과가 비어있는 경우
    if not output_path and not analysis_result:
        generation = """
이미지 분석 결과를 생성하지 못했습니다.

가능한 원인:
1. 이미지 파일 형식이 지원되지 않을 수 있습니다 (지원 형식: PNG, JPG, TIFF)
2. 이미지 경로가 올바르지 않을 수 있습니다
3. AI 모델 처리 중 오류가 발생했을 수 있습니다

이미지 경로를 확인하고 다시 시도해주세요.
"""
        return {"generation": generation, "previous_question": question}
    
    # 정상 처리 - LLM으로 친절한 설명 생성
    summary_prefix = f"[이전 대화 요약]\n{summary}\n\n" if summary else ""
    final_prompt = f"""
{summary_prefix}사용자 질문: {question}

AI 분석 결과:
{analysis_result}

출력 이미지 경로: {output_path}

위 결과를 바탕으로 사용자에게 간결하고 명확하게 설명해주세요.
분석 결과의 핵심 내용과 출력 이미지 경로를 알려주세요.
"""
    
    response = llm_vision.invoke(final_prompt)
    generation = response.content if hasattr(response, 'content') else str(response)
    
    # 이미지 경로를 generation에 명시적으로 포함
    if output_path:
        generation += f"\n\n📁 출력 이미지 경로: {output_path}"
    
    return {"generation": generation, "previous_question": question}
