#!/usr/bin/env python3
"""
데이터 로딩 및 전처리 모듈
S1 SAR 데이터 로딩, 전처리, 정규화 기능
"""

import torch
import numpy as np
import rasterio
from PIL import Image
from pathlib import Path
from typing import Tuple, List

def load_s1_sar_image(image_path: str) -> Tuple[np.ndarray, str]:
    """
    S1 SAR 이미지 로드 (GeoTIFF 또는 일반 이미지)
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        image_array: 이미지 배열 [C, H, W]
        file_type: 파일 타입 ("geotiff" 또는 "image")
    """
    try:
        # 먼저 rasterio로 시도 (GeoTIFF 파일)
        with rasterio.open(image_path) as src:
            image_array = src.read()  # [C, H, W] 순서로 읽음
            print(f"GeoTIFF 파일 로드됨: {image_path}, shape: {image_array.shape}")
            return image_array, "geotiff"
    except Exception as e:
        print(f"Rasterio로 읽기 실패, PIL로 시도: {e}")
        # PIL로 fallback
        image = Image.open(image_path).convert('RGB')
        image = image.resize((256, 256))
        image_array = np.array(image)
        # RGB를 2채널로 변환
        image_array = image_array[:2] if image_array.shape[2] >= 2 else image_array
        print(f"PIL로 로드됨: {image_path}, shape: {image_array.shape}")
        return image_array, "image"

def preprocess_s1_sar_data(image_array: np.ndarray, target_size: int = 256) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    S1 SAR 데이터 전처리 및 정규화
    
    Args:
        image_array: 원본 이미지 배열 [C, H, W]
        target_size: 목표 이미지 크기
        
    Returns:
        image_tensor: 전처리된 텐서 [1, 2, H, W]
        vv_band: VV 채널 (시각화용)
        vh_band: VH 채널 (시각화용)
    """
    # S1 SAR 데이터 검증
    if image_array.shape[0] < 2:
        raise ValueError(f"S1 SAR 데이터는 최소 2채널(VV, VH)이 필요합니다. 현재 채널 수: {image_array.shape[0]}")
    
    # VV, VH 채널만 사용 (첫 2채널)
    image_array = image_array[:2]  # [2, H, W]
    
    # S1 SAR 정규화 (cobench_dfc2020_s1.yaml 기반)
    band_stats = {
        'mean': [-12.548, -20.192],
        'std': [5.257, 5.912]
    }
    
    # S1 SAR 정규화 적용
    img_bands = []
    for b in range(2):
        img = torch.from_numpy(image_array[b]).float()
        
        # H, W 차원을 256x256으로 리사이즈
        if img.shape[0] != target_size or img.shape[1] != target_size:
            # 간단한 리사이즈 (더 정교한 방법 필요시 cv2 사용)
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0).unsqueeze(0), 
                size=(target_size, target_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0).squeeze(0)
        
        # Quantile 기반 outlier 제거 (99%, 1%)
        max_q = torch.quantile(img.reshape(-1), 0.99)
        min_q = torch.quantile(img.reshape(-1), 0.01)
        img = torch.clamp(img, min_q, max_q)
        
        # 정규화
        mean = band_stats['mean'][b]
        std = band_stats['std'][b]
        min_value = mean - 2 * std
        max_value = mean + 2 * std
        img = (img - min_value) / (max_value - min_value)
        img = torch.clamp(img, 0, 1)
        img_bands.append(img)
    
    # 텐서로 변환
    image_tensor = torch.stack(img_bands, dim=0).unsqueeze(0)  # [1, 2, 256, 256]
    
    # 시각화용 데이터 (numpy)
    vv_band = img_bands[0].numpy()  # VV
    vh_band = img_bands[1].numpy()  # VH
    
    return image_tensor, vv_band, vh_band

def validate_s1_sar_data(image_array: np.ndarray) -> bool:
    """
    S1 SAR 데이터 유효성 검사
    
    Args:
        image_array: 이미지 배열
        
    Returns:
        bool: 유효한 데이터인지 여부
    """
    if len(image_array.shape) != 3:
        print(f"잘못된 차원 수: {len(image_array.shape)}, 예상: 3 [C, H, W]")
        return False
    
    if image_array.shape[0] < 2:
        print(f"채널 수 부족: {image_array.shape[0]}, 최소 필요: 2")
        return False
    
    if image_array.shape[1] < 16 or image_array.shape[2] < 16:
        print(f"이미지 크기 너무 작음: {image_array.shape[1]}x{image_array.shape[2]}, 최소: 16x16")
        return False
    
    return True

def load_s2_optical_image(s2_path: str) -> Tuple[np.ndarray, str]:
    """
    S2 Optical 데이터만 로드
    
    Args:
        s2_path: S2 Optical 이미지 경로
        
    Returns:
        image_array: S2 이미지 배열 [13, H, W]
        file_type: "optical"
    """
    with rasterio.open(s2_path) as src:
        s2_data = src.read()  # 모든 밴드 사용
        print(f"S2 Optical 로드: {s2_data.shape}")
    
    return s2_data, "optical"

def preprocess_s2_optical_data(image_array: np.ndarray, target_size: int = 256) -> torch.Tensor:
    """
    S2 Optical 데이터 전처리 및 정규화
    
    Args:
        image_array: 원본 이미지 배열 [13, H, W]
        target_size: 목표 이미지 크기
        
    Returns:
        image_tensor: 전처리된 텐서 [1, 13, H, W]
    """
    # S2 정규화 통계 (Sentinel-2 13개 밴드)
    s2_stats = {
        'mean': [1353.036, 1265.71, 1269.61, 1399.096, 1627.916, 2060.956, 2337.466, 
                 2345.366, 2484.336, 714.836, 16.526, 2046.976, 1461.056],
        'std': [65.479, 154.008, 187.997, 278.508, 228.122, 356.598, 456.035, 
                531.820, 548.565, 78.595, 4.929, 403.895, 340.681]
    }
    
    img_bands = []
    
    # 13개 밴드 모두 정규화
    for b in range(min(13, image_array.shape[0])):
        # numpy array를 float32로 변환 후 torch tensor로 변환
        img = torch.from_numpy(image_array[b].astype(np.float32))
        
        # 크기 조정
        if img.shape[0] != target_size or img.shape[1] != target_size:
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0).unsqueeze(0), 
                size=(target_size, target_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0).squeeze(0)
        
        # Quantile 기반 outlier 제거
        max_q = torch.quantile(img.reshape(-1), 0.99)
        min_q = torch.quantile(img.reshape(-1), 0.01)
        img = torch.clamp(img, min_q, max_q)
        
        # 정규화
        mean = s2_stats['mean'][b]
        std = s2_stats['std'][b]
        min_value = mean - 2 * std
        max_value = mean + 2 * std
        img = (img - min_value) / (max_value - min_value)
        img = torch.clamp(img, 0, 1)
        img_bands.append(img)
    
    # 텐서로 변환
    image_tensor = torch.stack(img_bands, dim=0).unsqueeze(0)  # [1, 13, 256, 256]
    
    return image_tensor

def load_s1_s2_fusion_image(s1_path: str, s2_path: str = None) -> Tuple[np.ndarray, str]:
    """
    S1과 S2 데이터를 함께 로드 (Fusion)
    
    Args:
        s1_path: S1 SAR 이미지 경로
        s2_path: S2 Optical 이미지 경로 (None이면 s1_path에서 자동 추론)
        
    Returns:
        image_array: 통합 이미지 배열 [15, H, W] (S1: 2채널 + S2: 13채널)
        file_type: "fusion"
    """
    # S1 데이터 로드
    with rasterio.open(s1_path) as src:
        s1_data = src.read()[:2]  # VV, VH만 사용
        print(f"S1 로드: {s1_data.shape}")
    
    # S2 경로 자동 추론
    if s2_path is None:
        s2_path = s1_path.replace('/s1/', '/s2/').replace('_s1_', '_s2_')
        print(f"S2 경로 자동 추론: {s2_path}")
    
    # S2 데이터 로드
    with rasterio.open(s2_path) as src:
        s2_data = src.read()  # 모든 밴드 사용
        print(f"S2 로드: {s2_data.shape}")
    
    # S1과 S2 크기 맞추기 (256x256)
    target_size = 256
    
    # S1 리사이즈
    if s1_data.shape[1] != target_size or s1_data.shape[2] != target_size:
        s1_resized = []
        for i in range(s1_data.shape[0]):
            s1_band = torch.from_numpy(s1_data[i]).float()
            s1_band = torch.nn.functional.interpolate(
                s1_band.unsqueeze(0).unsqueeze(0), 
                size=(target_size, target_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze().numpy()
            s1_resized.append(s1_band)
        s1_data = np.array(s1_resized)
    
    # S2 리사이즈
    if s2_data.shape[1] != target_size or s2_data.shape[2] != target_size:
        s2_resized = []
        for i in range(s2_data.shape[0]):
            s2_band = torch.from_numpy(s2_data[i]).float()
            s2_band = torch.nn.functional.interpolate(
                s2_band.unsqueeze(0).unsqueeze(0), 
                size=(target_size, target_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze().numpy()
            s2_resized.append(s2_band)
        s2_data = np.array(s2_resized)
    
    # S1 + S2 concatenate
    fusion_data = np.concatenate([s1_data, s2_data], axis=0)  # [15, 256, 256]
    print(f"Fusion 데이터: {fusion_data.shape}")
    
    return fusion_data, "fusion"

def preprocess_s1_s2_fusion_data(image_array: np.ndarray, target_size: int = 256) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    S1+S2 Fusion 데이터 전처리 및 정규화
    
    Args:
        image_array: 원본 이미지 배열 [15, H, W]
        target_size: 목표 이미지 크기
        
    Returns:
        image_tensor: 전처리된 텐서 [1, 15, H, W]
        vv_band: VV 채널 (시각화용)
        vh_band: VH 채널 (시각화용)
    """
    # S1 정규화 (처음 2채널)
    s1_stats = {
        'mean': [-12.548, -20.192],
        'std': [5.257, 5.912]
    }
    
    # S2 정규화 (나머지 13채널) - Sentinel-2 통계
    s2_stats = {
        'mean': [1353.036, 1265.71, 1269.61, 1399.096, 1627.916, 2060.956, 2337.466, 
                 2345.366, 2484.336, 714.836, 16.526, 2046.976, 1461.056],
        'std': [65.479, 154.008, 187.997, 278.508, 228.122, 356.598, 456.035, 
                531.820, 548.565, 78.595, 4.929, 403.895, 340.681]
    }
    
    img_bands = []
    
    # S1 정규화 (채널 0, 1)
    for b in range(2):
        img = torch.from_numpy(image_array[b]).float()
        
        # Quantile 기반 outlier 제거
        max_q = torch.quantile(img.reshape(-1), 0.99)
        min_q = torch.quantile(img.reshape(-1), 0.01)
        img = torch.clamp(img, min_q, max_q)
        
        # 정규화
        mean = s1_stats['mean'][b]
        std = s1_stats['std'][b]
        min_value = mean - 2 * std
        max_value = mean + 2 * std
        img = (img - min_value) / (max_value - min_value)
        img = torch.clamp(img, 0, 1)
        img_bands.append(img)
    
    # S2 정규화 (채널 2~14)
    for b in range(13):
        img = torch.from_numpy(image_array[b + 2]).float()
        
        # Quantile 기반 outlier 제거
        max_q = torch.quantile(img.reshape(-1), 0.99)
        min_q = torch.quantile(img.reshape(-1), 0.01)
        img = torch.clamp(img, min_q, max_q)
        
        # 정규화
        mean = s2_stats['mean'][b]
        std = s2_stats['std'][b]
        min_value = mean - 2 * std
        max_value = mean + 2 * std
        img = (img - min_value) / (max_value - min_value)
        img = torch.clamp(img, 0, 1)
        img_bands.append(img)
    
    # 텐서로 변환
    image_tensor = torch.stack(img_bands, dim=0).unsqueeze(0)  # [1, 15, 256, 256]
    
    # 시각화용 데이터 (numpy) - S1 VV, VH
    vv_band = img_bands[0].numpy()
    vh_band = img_bands[1].numpy()
    
    return image_tensor, vv_band, vh_band

def get_image_metadata(image_path: str) -> dict:
    """
    이미지 메타데이터 추출
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        dict: 메타데이터 정보
    """
    metadata = {
        "file_path": image_path,
        "file_name": Path(image_path).name,
        "file_size": Path(image_path).stat().st_size if Path(image_path).exists() else 0
    }
    
    try:
        with rasterio.open(image_path) as src:
            # 좌표 범위 계산
            bounds = src.bounds  # (minx, miny, maxx, maxy)
            
            # 기본 이미지 정보
            basic_info = {
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": str(src.dtypes[0]),
                "crs": str(src.crs) if src.crs else "Unknown",
                "transform": str(src.transform),
                "bounds": {
                    "minx": float(bounds.left),
                    "miny": float(bounds.bottom),
                    "maxx": float(bounds.right),
                    "maxy": float(bounds.top)
                },
                "center": {
                    "x": float((bounds.left + bounds.right) / 2),
                    "y": float((bounds.bottom + bounds.top) / 2)
                },
                "spatial_resolution": {
                    "x": float(abs(src.transform[0])),
                    "y": float(abs(src.transform[4]))
                }
            }
            
            # 모든 태그 정보 추출
            tags_info = {}
            for i in range(src.count):
                band_tags = src.tags(i + 1)  # 밴드는 1부터 시작
                if band_tags:
                    tags_info[f"band_{i+1}"] = dict(band_tags)
            
            # 전역 태그 정보
            global_tags = dict(src.tags())
            
            # 날짜/시간 정보 추출 (다양한 태그에서 시도)
            date_info = {}
            date_tags = ['TIFFTAG_DATETIME', 'datetime', 'DATE', 'ACQUISITION_DATE', 
                        'SENSING_TIME', 'ACQUISITION_TIME', 'PRODUCT_TIME', 'CREATION_TIME']
            
            for tag in date_tags:
                if tag in global_tags:
                    date_info[tag.lower()] = global_tags[tag]
                # 밴드별로도 확인
                for band_num in range(src.count):
                    band_tags = src.tags(band_num + 1)
                    if tag in band_tags:
                        date_info[f"band_{band_num+1}_{tag.lower()}"] = band_tags[tag]
            
            # 센서/위성 정보 추출
            sensor_info = {}
            sensor_tags = ['SATELLITE', 'SENSOR', 'PLATFORM', 'MISSION', 'INSTRUMENT',
                          'SATELLITE_ID', 'SENSOR_ID', 'MISSION_ID']
            
            for tag in sensor_tags:
                if tag in global_tags:
                    sensor_info[tag.lower()] = global_tags[tag]
            
            # 처리 정보 추출
            processing_info = {}
            processing_tags = ['PROCESSING_LEVEL', 'PRODUCT_TYPE', 'PRODUCT_ID', 
                             'PROCESSING_CENTER', 'SOFTWARE_VERSION', 'ALGORITHM_VERSION']
            
            for tag in processing_tags:
                if tag in global_tags:
                    processing_info[tag.lower()] = global_tags[tag]
            
            # 모든 정보 통합
            metadata.update(basic_info)
            metadata.update({
                "tags": {
                    "global": global_tags,
                    "bands": tags_info
                },
                "date_time": date_info,
                "sensor": sensor_info,
                "processing": processing_info,
                "all_tags": global_tags  # 모든 태그를 한 번에 보기 위해
            })
            
    except Exception as e:
        metadata["error"] = str(e)
    
    return metadata
