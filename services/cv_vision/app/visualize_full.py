"""
S1, S2, Prediction mask를 함께 시각화하는 모듈
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from PIL import Image
import os

# DFC2020 클래스 컬러맵
DFC_COLORMAP = {
    0: [0, 0, 0],        # Background
    1: [34, 139, 34],    # Forest
    2: [107, 142, 35],   # Shrubland
    3: [154, 205, 50],   # Grassland
    4: [0, 191, 255],    # Wetland
    5: [255, 165, 0],    # Cropland
    6: [128, 128, 128],  # Urban
    7: [139, 69, 19],    # Barren
    8: [0, 0, 255],      # Water
}

def normalize_image(img, percentile_range=(2, 98)):
    img = img.astype(float)
    img = np.where(np.isfinite(img), img, 0)
    if np.any(img > 0):
        img_min, img_max = np.percentile(img[img > 0], percentile_range)
    else:
        img_min, img_max = 0, 1
    if img_max > img_min:
        img = np.clip((img - img_min) / (img_max - img_min) * 255, 0, 255)
    else:
        img = np.zeros_like(img)
    return img.astype(np.uint8)

def load_s1_rgb(s1_path):
    """S1 SAR 이미지를 RGB로 변환"""
    with rasterio.open(s1_path) as src:
        vv = src.read(1)
        vh = src.read(2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = vv / vh
        ratio = np.where(np.isfinite(ratio), ratio, 0)
    
    r = normalize_image(vv)
    g = normalize_image(vh)
    b = normalize_image(ratio)
    
    return np.stack([r, g, b], axis=-1)

def load_s2_rgb(s2_path):
    """S2 Optical 이미지를 RGB로 변환"""
    with rasterio.open(s2_path) as src:
        red = src.read(4)
        green = src.read(3)
        blue = src.read(2)
    
    r = normalize_image(red)
    g = normalize_image(green)
    b = normalize_image(blue)
    
    return np.stack([r, g, b], axis=-1)

def create_full_visualization(s1_path, s2_path, mask_path, output_path, lulc_stats=None):
    """
    S1 SAR, S2 Optical, Prediction mask를 함께 시각화
    
    Args:
        s1_path: S1 SAR 이미지 경로
        s2_path: S2 Optical 이미지 경로 (자동 추론 가능)
        mask_path: Prediction mask 이미지 경로
        output_path: 출력 파일 경로
        lulc_stats: LULC 통계 딕셔너리 (선택)
    """
    # S2 경로 자동 추론
    if s2_path is None:
        s2_path = s1_path.replace('/s1/', '/s2/').replace('_s1_', '_s2_')
    
    # 데이터 로드
    s1_rgb = load_s1_rgb(s1_path)
    
    if os.path.exists(s2_path):
        s2_rgb = load_s2_rgb(s2_path)
        has_s2 = True
    else:
        s2_rgb = None
        has_s2 = False
    
    mask_img = Image.open(mask_path)
    mask = np.array(mask_img)
    
    # 시각화
    if has_s2:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # S1 SAR
        axes[0].imshow(s1_rgb)
        axes[0].set_title('S1 SAR\n(VV, VH, VV/VH)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # S2 Optical
        axes[1].imshow(s2_rgb)
        axes[1].set_title('S2 Optical\n(RGB True Color)', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Prediction
        axes[2].imshow(mask)
        if lulc_stats:
            top3 = '\n'.join([f'{k}: {v["percentage"]:.1f}%' 
                            for k, v in sorted(lulc_stats.items(), 
                                             key=lambda x: -x[1]["percentage"])[:3]])
            axes[2].set_title(f'LULC Prediction\n{top3}', fontsize=12, fontweight='bold')
        else:
            axes[2].set_title('LULC Prediction', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        # 범례 추가
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=np.array(color)/255, label=label) 
                          for label, color in {
                              'Forest': [34, 139, 34],
                              'Shrubland': [107, 142, 35],
                              'Grassland': [154, 205, 50],
                              'Wetland': [0, 191, 255],
                              'Cropland': [255, 165, 0],
                              'Urban': [128, 128, 128],
                              'Barren': [139, 69, 19],
                              'Water': [0, 0, 255]
                          }.items()]
        axes[2].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        
    else:
        # S2 없으면 S1과 Prediction만
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].imshow(s1_rgb)
        axes[0].set_title('S1 SAR\n(VV, VH, VV/VH)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(mask)
        if lulc_stats:
            top3 = '\n'.join([f'{k}: {v["percentage"]:.1f}%' 
                            for k, v in sorted(lulc_stats.items(), 
                                             key=lambda x: -x[1]["percentage"])[:3]])
            axes[1].set_title(f'LULC Prediction\n{top3}', fontsize=12, fontweight='bold')
        else:
            axes[1].set_title('LULC Prediction', fontsize=14, fontweight='bold')
        axes[1].axis('off')
    
    plt.suptitle('SAR Image Analysis Result', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path
