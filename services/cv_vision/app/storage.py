import os, json
from pathlib import Path
import numpy as np
from PIL import Image
from datetime import datetime
import uuid
from typing import Optional
from app.paths import NAS_ROOT

# 로컬 output 폴더에 직접 저장
OUTPUT_DIR = Path(NAS_ROOT)  # /home/mjh/Project/LLM/RAG/ai-service/output

def generate_filename(job_id: str, task_type: str, file_type: str, 
                     extension: str, custom_name: Optional[str] = None,
                     sar_image_id: Optional[str] = None,
                     request_id: Optional[str] = None) -> str:
    """파일명 생성 함수"""
    if custom_name:
        # 사용자 정의 파일명 사용
        return f"{custom_name}.{extension}"
    
    if sar_image_id and request_id:
        # 시간_sar영상ID_요청ID 형식 (기본)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]  # 8자리 UUID
        return f"{timestamp}_{sar_image_id}_{request_id}_{unique_id}.{extension}"
    else:
        # 기본 타임스탬프 기반 파일명
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{task_type}_{file_type}.{extension}"

def get_output_directory(user_id: Optional[str] = None, request_id: Optional[str] = None,
                        output_folder: Optional[str] = None) -> Path:
    """출력 디렉토리 생성"""
    if output_folder:
        # 사용자 정의 폴더 경로 사용
        output_dir = OUTPUT_DIR / output_folder
    elif user_id and request_id:
        # 사용자ID/요청ID 구조
        output_dir = OUTPUT_DIR / user_id / request_id
    else:
        # 기본 출력 디렉토리
        output_dir = OUTPUT_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def save_mask_png(job_id: str, mask: np.ndarray, 
                  custom_name: Optional[str] = None,
                  user_id: Optional[str] = None,
                  request_id: Optional[str] = None,
                  sar_image_id: Optional[str] = None,
                  output_folder: Optional[str] = None) -> str:
    """마스크 PNG 저장 (컬러 팔레트 적용)"""
    # 클래스별 컬러 매핑 (S1 SAR 데이터용)
    class_colors = {
        0: [0, 0, 0],        # Background - 검은색
        1: [34, 139, 34],    # Forest - 녹색
        2: [107, 142, 35],   # Shrubland - 올리브색
        3: [154, 205, 50],   # Grassland - 연두색
        4: [0, 191, 255],    # Wetland - 하늘색
        5: [255, 165, 0],    # Cropland - 주황색
        6: [128, 128, 128],  # Urban/Built-up - 회색
        7: [139, 69, 19],    # Barren - 갈색
        8: [0, 0, 255],      # Water - 파란색
    }
    
    # 컬러 마스크 생성
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in class_colors.items():
        mask_class = (mask == class_id)
        color_mask[mask_class] = color
    
    output_dir = get_output_directory(user_id, request_id, output_folder)
    filename = generate_filename(job_id, "segmentation", "mask", "png", 
                               custom_name, sar_image_id, request_id)
    out = output_dir / filename
    Image.fromarray(color_mask).save(out)
    return str(out)

def save_mask_tif(job_id: str, mask: np.ndarray, 
                  custom_name: Optional[str] = None,
                  user_id: Optional[str] = None,
                  request_id: Optional[str] = None,
                  sar_image_id: Optional[str] = None,
                  output_folder: Optional[str] = None) -> str:
    """마스크 GeoTIFF 저장"""
    output_dir = get_output_directory(user_id, request_id, output_folder)
    filename = generate_filename(job_id, "segmentation", "mask", "tif", 
                               custom_name, sar_image_id, request_id)
    out = output_dir / filename
    # GeoTIFF 저장 로직 (rasterio 사용)
    return str(out)

def save_json(job_id: str, data: dict, filename: str = "result",
              custom_name: Optional[str] = None,
              user_id: Optional[str] = None,
              request_id: Optional[str] = None,
              sar_image_id: Optional[str] = None,
              output_folder: Optional[str] = None) -> str:
    """JSON 결과 저장"""
    output_dir = get_output_directory(user_id, request_id, output_folder)
    
    if custom_name:
        out_filename = f"{custom_name}.json"
    elif sar_image_id and request_id:
        # 시간_sar영상ID_요청ID 형식 (기본)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        out_filename = f"{timestamp}_{sar_image_id}_{request_id}_{unique_id}.json"
    else:
        # 기본 타임스탬프 기반 파일명
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_filename = f"{timestamp}_{filename}.json"
    
    out = output_dir / out_filename
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(out)

def save_image(job_id: str, image_array: np.ndarray, file_type: str,
               custom_name: Optional[str] = None,
               user_id: Optional[str] = None,
               request_id: Optional[str] = None,
               sar_image_id: Optional[str] = None,
               output_folder: Optional[str] = None) -> str:
    """일반 이미지 저장 (PNG, JPG 등)"""
    output_dir = get_output_directory(user_id, request_id, output_folder)
    filename = generate_filename(job_id, "visualization", file_type, 
                               file_type, custom_name, sar_image_id, request_id)
    out = output_dir / filename
    Image.fromarray(image_array.astype("uint8")).save(out)
    return str(out)