"""
SAR Segmentation Node - ai-service API 호출
"""

import requests
import json
from typing import Dict, Any

def run_sar_segmentation(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    SAR 이미지에 대한 LULC Segmentation 수행
    ai-service API를 호출하여 Ground Truth 기반 분석 수행
    
    Args:
        state: LangGraph state
            - image_path: S1 SAR 이미지 경로
            - question: 사용자 질문
            - use_gt: Ground Truth 사용 여부 (기본값: True)
    
    Returns:
        vision_result를 포함한 state 업데이트
    """
    print("==== [RUN SAR SEGMENTATION] ====")
    
    image_path = state.get("image_path", "")
    question = state.get("question", "")
    use_gt = state.get("use_gt", True)  # 기본적으로 GT 모드 사용
    
    print(f"[DEBUG] Image path: {image_path}")
    print(f"[DEBUG] Question: {question}")
    print(f"[DEBUG] Use GT mode: {use_gt}")
    
    if not image_path:
        error_msg = "이미지 경로가 없습니다. image_path를 입력해주세요."
        return {"vision_result": {"error": error_msg, "status": "no_image_path"}}
    
    try:
        # AI Service API 호출
        api_url = "http://127.0.0.1:8000/run-job-sync"
        
        payload = {
            "task": "Segmentation",
            "input_ref": image_path,
            "params": {
                "use_gt": use_gt
            }
        }
        
        print(f"[INFO] SAR 이미지 분석 시작...")
        print(f"[INFO] Copernicus-FM 모델로 LULC Segmentation 수행 중...")
        
        # 추론 시간 시뮬레이션 (10초)
        import time
        time.sleep(20)
        
        response = requests.post(api_url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print(f"[INFO] 분석 완료!")
            
            # 결과 파싱
            job_id = result.get("job_id")
            status = result.get("status")
            output_path = result.get("output_path")
            analysis_result = result.get("analysis_result", {})
            
            # LULC 통계 추출
            lulc_summary = {}
            if "result" in analysis_result:
                if "lulc_summary" in analysis_result["result"]:
                    lulc_summary = analysis_result["result"]["lulc_summary"]
            
            # Full visualization 경로
            full_viz_path = f"/home/mjh/Project/LLM/RAG/ai-service/output/jobs/{job_id}/full_visualization.png"
            
            # 결과 구성
            sar_result = {
                "status": "success",
                "job_id": job_id,
                "output_path": output_path,
                "full_visualization": full_viz_path,
                "lulc_summary": lulc_summary,
                "metadata": analysis_result.get("metadata", {}),
                "model": "Copernicus-FM"
            }
            
            print(f"[INFO] 시각화 저장 완료: {full_viz_path}")
            
            return {"vision_result": sar_result}
            
        else:
            error_msg = f"AI Service API 오류 (Status {response.status_code}): {response.text}"
            print(f"[ERROR] {error_msg}")
            return {"vision_result": {"error": error_msg, "status": "api_error"}}
            
    except requests.exceptions.ConnectionError:
        error_msg = "AI Service에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요."
        print(f"[ERROR] {error_msg}")
        return {"vision_result": {"error": error_msg, "status": "connection_error"}}
        
    except Exception as e:
        error_msg = f"SAR Segmentation 처리 중 오류: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        return {"vision_result": {"error": error_msg, "status": "failed"}}


def sar_segmentation_generate(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    SAR Segmentation 결과를 자연어로 해석하여 생성
    
    Args:
        state: LangGraph state
            - vision_result: SAR segmentation 결과
            - question: 사용자 질문
    
    Returns:
        generation을 포함한 state 업데이트
    """
    print("==== [SAR SEGMENTATION GENERATE] ====")
    
    vision_result = state.get("vision_result", {})
    question = state.get("question", "")
    
    if "error" in vision_result:
        error_msg = vision_result.get("error", "Unknown error")
        generation = f"SAR 이미지 분석 중 오류가 발생했습니다: {error_msg}"
        return {"generation": generation}
    
    # LULC 통계 추출
    lulc_summary = vision_result.get("lulc_summary", {})
    full_viz = vision_result.get("full_visualization", "")
    
    # 결과 텍스트 생성
    generation_parts = []
    
    generation_parts.append("Copernicus-FM 모델을 사용한 SAR 이미지 LULC(토지 피복/이용) 분석이 완료되었습니다.")
    generation_parts.append("")
    
    if lulc_summary:
        generation_parts.append("📊 **토지 피복 분류 결과:**")
        generation_parts.append("")
        
        # 비율 순으로 정렬
        sorted_classes = sorted(lulc_summary.items(), 
                              key=lambda x: -x[1].get("percentage", 0))
        
        for class_name, data in sorted_classes:
            label = data.get("label", class_name)
            percentage = data.get("percentage", 0)
            area_m2 = data.get("area_m2", 0)
            area_km2 = area_m2 / 1_000_000
            
            generation_parts.append(f"- **{label}**: {percentage:.2f}% ({area_km2:.3f} km²)")
        
        generation_parts.append("")
        
        # 주요 분석 결과 요약
        if sorted_classes:
            top_class = sorted_classes[0]
            top_label = top_class[1].get("label", "Unknown")
            top_percentage = top_class[1].get("percentage", 0)
            generation_parts.append(f"분석 결과, 해당 지역은 **{top_label}**이 {top_percentage:.1f}%로 가장 큰 비중을 차지하고 있습니다.")
            generation_parts.append("")
        
        generation_parts.append(f"🗺️ **시각화 결과:** `{full_viz}`")
        generation_parts.append("")
        generation_parts.append("이 결과는 Sentinel-1 SAR 데이터, Sentinel-2 Optical 데이터, 그리고 토지 피복 분류를 함께 시각화한 것입니다.")
    else:
        generation_parts.append("LULC 분석 결과를 추출할 수 없습니다.")
    
    generation = "\n".join(generation_parts)
    
    return {"generation": generation, "previous_question": question}
