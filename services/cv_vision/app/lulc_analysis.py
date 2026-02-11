#!/usr/bin/env python3
"""
LULC (Land Use Land Cover) 분석 도구
Segmentation 결과를 기반으로 LULC 통계 및 JSON 생성
"""

import numpy as np
from datetime import datetime
from typing import Dict, Any

def create_analysis_result(job_id: str, task: str, result_data: Dict[str, Any], input_ref: str, 
                          additional_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """범용 분석 결과 JSON 생성 (모든 task 공통)"""
    from app.load import get_image_metadata
    
    # 이미지 메타데이터 추출
    image_metadata = get_image_metadata(input_ref)
    
    # 현재 날짜 및 시간
    current_datetime = datetime.now()
    current_date = current_datetime.strftime("%Y-%m-%d")
    current_time = current_datetime.strftime("%H:%M:%S")
    
    # AOI ID 생성
    aoi_id = f"Area_{job_id}_{current_date}"
    
    # 기본 메타데이터
    base_metadata = {
        "job_id": job_id,
        "task": task,
        "aoi_id": aoi_id,
        "timestamp": {
            "date": current_date,
            "time": current_time,
            "iso": current_datetime.isoformat()
        },
        "input_file": {
            "path": input_ref,
            "filename": image_metadata.get("file_name", "unknown"),
            "file_size_bytes": image_metadata.get("file_size", 0)
        },
        "image_metadata": {
            "width": image_metadata.get("width", 0),
            "height": image_metadata.get("height", 0),
            "channels": image_metadata.get("count", 0),
            "data_type": image_metadata.get("dtype", "unknown"),
            "crs": image_metadata.get("crs", "Unknown"),
            "transform": str(image_metadata.get("transform", "Unknown")),
            "bounds": image_metadata.get("bounds", {}),
            "center": image_metadata.get("center", {}),
            "spatial_resolution": image_metadata.get("spatial_resolution", {}),
            "date_time": image_metadata.get("date_time", {}),
            "sensor": image_metadata.get("sensor", {}),
            "processing": image_metadata.get("processing", {}),
            "tags": image_metadata.get("tags", {}),
            "all_tags": image_metadata.get("all_tags", {})
        },
        "processing_info": {
            "model": "Copernicus-FM" if task == "Segmentation" else "DualTaskWrapper",
            "framework": "PyTorch",
            "api_version": "1.0.0"
        }
    }
    
    # 추가 메타데이터가 있으면 병합
    if additional_metadata:
        base_metadata.update(additional_metadata)
    
    # 결과 데이터와 메타데이터 결합
    analysis_result = {
        "metadata": base_metadata,
        "result": result_data
    }
    
    return analysis_result

def create_lulc_analysis(job_id: str, predicted_mask: np.ndarray, input_ref: str) -> Dict[str, Any]:
    """LULC 분석 결과 JSON 생성 (기존 호환성 유지)"""
    
    # 클래스 매핑 (S1 SAR 데이터용)
    class_mapping = {
        0: {"id": 0, "label": "Background", "color": "black"},
        1: {"id": 1, "label": "Forest", "color": "green"},
        2: {"id": 2, "label": "Shrubland", "color": "olive"},
        3: {"id": 3, "label": "Grassland", "color": "lightgreen"},
        4: {"id": 4, "label": "Wetland", "color": "skyblue"},
        5: {"id": 5, "label": "Cropland", "color": "orange"},
        6: {"id": 6, "label": "Urban/Built-up", "color": "gray"},
        7: {"id": 7, "label": "Barren", "color": "brown"},
        8: {"id": 8, "label": "Water", "color": "blue"}
    }
    
    # 마스크 크기 및 픽셀 수 계산
    h, w = predicted_mask.shape
    total_pixels = h * w
    
    # 픽셀 크기 (10m x 10m = 100m²)
    pixel_area_m2 = 100
    
    # 클래스별 통계 계산
    lulc_summary = {}
    total_area_m2 = 0
    
    for class_id, class_info in class_mapping.items():
        mask_class = (predicted_mask == class_id)
        pixel_count = np.sum(mask_class)
        
        if pixel_count > 0:
            area_m2 = pixel_count * pixel_area_m2
            percentage = (pixel_count / total_pixels) * 100
            total_area_m2 += area_m2
            
            lulc_summary[class_info["label"].lower().replace("/", "_").replace("-", "_")] = {
                "class_id": class_id,
                "label": class_info["label"],
                "area_m2": int(area_m2),
                "percentage": round(percentage, 2)
            }
    
    # LULC 전용 메타데이터
    lulc_metadata = {
        "analysis_type": "Land Use Land Cover",
        "source": "Sentinel-1 SAR",
        "resolution": "10m",
        "projection": "EPSG:4326",
        "image_size": f"{h}x{w}",
        "total_pixels": int(total_pixels),
        "total_area_m2": int(total_area_m2),
        "change_analysis": {
            "from": None,
            "to": datetime.now().strftime("%Y-%m-%d"),
            "note": "Baseline analysis - no previous data for comparison"
        }
    }
    
    # LULC 결과 데이터
    lulc_data = {
        "lulc_summary": lulc_summary
    }
    
    # 범용 분석 결과 생성
    return create_analysis_result(job_id, "Segmentation", lulc_data, input_ref, lulc_metadata)

def calculate_class_statistics(predicted_mask: np.ndarray) -> Dict[str, Any]:
    """클래스별 통계 계산 (면적, 비율 등)"""
    
    class_mapping = {
        0: "Background",
        1: "Forest",
        2: "Shrubland", 
        3: "Grassland",
        4: "Wetland",
        5: "Cropland",
        6: "Urban/Built-up",
        7: "Barren",
        8: "Water"
    }
    
    h, w = predicted_mask.shape
    total_pixels = h * w
    pixel_area_m2 = 100  # 10m x 10m
    
    statistics = {}
    
    for class_id, class_name in class_mapping.items():
        mask_class = (predicted_mask == class_id)
        pixel_count = np.sum(mask_class)
        
        if pixel_count > 0:
            area_m2 = pixel_count * pixel_area_m2
            percentage = (pixel_count / total_pixels) * 100
            
            statistics[class_name] = {
                "pixel_count": int(pixel_count),
                "area_m2": int(area_m2),
                "percentage": round(percentage, 2)
            }
    
    return statistics

def create_change_analysis(previous_mask: np.ndarray, current_mask: np.ndarray) -> Dict[str, Any]:
    """변화 분석 (이전 마스크와 현재 마스크 비교)"""
    
    # 클래스별 변화율 계산
    prev_stats = calculate_class_statistics(previous_mask)
    curr_stats = calculate_class_statistics(current_mask)
    
    change_rates = {}
    
    for class_name in prev_stats.keys():
        if class_name in curr_stats:
            prev_area = prev_stats[class_name]["area_m2"]
            curr_area = curr_stats[class_name]["area_m2"]
            
            if prev_area > 0:
                change_rate = ((curr_area - prev_area) / prev_area) * 100
                change_rates[f"{class_name.lower()}_change_rate"] = round(change_rate, 2)
    
    return {
        "change_rates": change_rates,
        "previous_stats": prev_stats,
        "current_stats": curr_stats
    }

