"""Vision Agent Tools 정의"""
from langchain_core.tools import tool
import requests
from requests.exceptions import ConnectionError, Timeout, RequestException

FASTAPI_BASE_URL = "http://192.168.10.174:6600"


@tool
def segmentation_tool(image_path: str) -> dict:
    """
    SAR 이미지에서 Land Use/Land Cover(LULC) Segmentation을 수행합니다.
    위성 이미지를 분석하여 건물, 도로, 농지, 수역 등을 분류합니다.
    
    Args:
        image_path: 분석할 SAR 이미지 경로 (예: /path/to/image.tif)
    
    Returns:
        LULC 분석 결과 (클래스별 면적, 비율, 통계 정보)
    """
    try:
        response = requests.post(
            f"{FASTAPI_BASE_URL}/run-job-sync",  
            json={
                "task": "Segmentation",
                "input_ref": image_path,
            },
            timeout=120  
        )
        response.raise_for_status()
        return response.json()
    except ConnectionError as e:
        return {
            "error": f"FastAPI 서버에 연결할 수 없습니다 ({FASTAPI_BASE_URL})",
            "details": "서버가 실행 중인지 확인해주세요",
            "status": "connection_error",
            "server_url": FASTAPI_BASE_URL
        }
    except Timeout as e:
        return {
            "error": "서버 응답 시간이 초과되었습니다",
            "details": "이미지 처리 시간이 너무 오래 걸립니다",
            "status": "timeout_error"
        }
    except RequestException as e:
        return {
            "error": f"요청 처리 중 오류가 발생했습니다: {str(e)}",
            "status": "request_error"
        }
    except Exception as e:
        return {
            "error": f"알 수 없는 오류가 발생했습니다: {str(e)}",
            "status": "unknown_error"
        }


@tool  
def classification_tool(image_path: str) -> dict:
    """
    이미지 분류를 수행합니다.
    이미지가 어떤 카테고리에 속하는지 판단합니다.
    
    Args:
        image_path: 분류할 이미지 경로
    
    Returns:
        분류 결과 (class_id, class_name, confidence)
    """
    try:
        response = requests.post(
            f"{FASTAPI_BASE_URL}/run-job-sync",  
            json={
                "task": "Classification",
                "input_ref": image_path,
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except ConnectionError as e:
        return {
            "error": f"FastAPI 서버에 연결할 수 없습니다 ({FASTAPI_BASE_URL})",
            "details": "서버가 실행 중인지 확인해주세요",
            "status": "connection_error",
            "server_url": FASTAPI_BASE_URL
        }
    except Timeout as e:
        return {
            "error": "서버 응답 시간이 초과되었습니다",
            "status": "timeout_error"
        }
    except RequestException as e:
        return {
            "error": f"요청 처리 중 오류가 발생했습니다: {str(e)}",
            "status": "request_error"
        }
    except Exception as e:
        return {
            "error": f"알 수 없는 오류가 발생했습니다: {str(e)}",
            "status": "unknown_error"
        }


@tool
def detection_tool(image_path: str, confidence_threshold: float = 0.5) -> dict:
    """
    이미지에서 객체를 검출합니다.
    선박, 차량 등의 객체 위치와 종류를 찾아냅니다.
    
    Args:
        image_path: 검출할 이미지 경로
        confidence_threshold: 검출 신뢰도 임계값 (기본 0.5)
    
    Returns:
        검출 결과 (detections 목록, num_detections, bounding boxes)
    """
    try:
        response = requests.post(
            f"{FASTAPI_BASE_URL}/run-job-sync",  
            json={
                "task": "Detection",
                "input_ref": image_path,
                "params": {"confidence_threshold": confidence_threshold}
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except ConnectionError as e:
        return {
            "error": f"FastAPI 서버에 연결할 수 없습니다 ({FASTAPI_BASE_URL})",
            "details": "서버가 실행 중인지 확인해주세요",
            "status": "connection_error",
            "server_url": FASTAPI_BASE_URL
        }
    except Timeout as e:
        return {
            "error": "서버 응답 시간이 초과되었습니다",
            "status": "timeout_error"
        }
    except RequestException as e:
        return {
            "error": f"요청 처리 중 오류가 발생했습니다: {str(e)}",
            "status": "request_error"
        }
    except Exception as e:
        return {
            "error": f"알 수 없는 오류가 발생했습니다: {str(e)}",
            "status": "unknown_error"
        }


# Vision Tools 리스트
vision_tools = [segmentation_tool, classification_tool, detection_tool]
