#!/usr/bin/env python3
"""
Segmentation 결과 시각화 도구
S1 SAR 데이터 segmentation 결과를 컬러 마스크로 변환
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
import argparse
import sys

def create_color_mask(mask: np.ndarray, output_path: str, output_format: str = "tif") -> str:
    """컬러 시각화 마스크 생성 및 저장"""
    from PIL import Image
    
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
    
    # 클래스 이름 매핑
    class_names = {
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
    
    # 컬러 마스크 생성
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    print(f"마스크 크기: {h}x{w}")
    print("클래스 분포:")
    
    for class_id, color in class_colors.items():
        mask_class = (mask == class_id)
        pixel_count = np.sum(mask_class)
        if pixel_count > 0:
            print(f"  클래스 {class_id} ({class_names[class_id]}): {pixel_count} 픽셀 ({pixel_count/(h*w)*100:.1f}%)")
            color_mask[mask_class] = color
    
    # 파일 형식에 따라 저장
    if output_format.lower() == "png":
        # PNG로 저장
        pil_image = Image.fromarray(color_mask)
        pil_image.save(output_path)
        print(f"컬러 마스크 저장됨 (PNG): {output_path}")
    else:
        # GeoTIFF로 저장
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=h,
            width=w,
            count=3,
            dtype=rasterio.uint8,
            crs='EPSG:4326',
            transform=from_bounds(0, 0, 1, 1, w, h)  # 임시 transform
        ) as dst:
            dst.write(color_mask.transpose(2, 0, 1))  # [C, H, W] 순서로 변환
        print(f"컬러 마스크 저장됨 (GeoTIFF): {output_path}")
    
    return output_path

def load_mask_from_png(png_path: str) -> np.ndarray:
    """PNG 마스크 파일 로드"""
    from PIL import Image
    
    image = Image.open(png_path)
    mask = np.array(image)
    
    # 그레이스케일인 경우
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]  # 첫 번째 채널만 사용
    
    return mask

def load_mask_from_tif(tif_path: str) -> np.ndarray:
    """GeoTIFF 마스크 파일 로드"""
    with rasterio.open(tif_path) as src:
        mask = src.read(1)  # 첫 번째 밴드만 읽기
    
    return mask

def create_3channel_visualization(vv_band: np.ndarray, vh_band: np.ndarray, output_path: str) -> str:
    """3채널 시각화 이미지 생성 (VV, VH, (VV+VH)/2)"""
    from PIL import Image
    
    # (VV+VH)/2 계산
    vv_vh_avg = (vv_band + vh_band) / 2
    
    # 3채널 이미지 스택
    image_3ch = np.stack([vv_band, vh_band, vv_vh_avg], axis=0)  # [3, H, W]
    
    # 정규화된 값을 0-255 범위로 변환
    image_3ch_vis = np.zeros((3, 256, 256), dtype=np.uint8)
    for i in range(3):
        # 각 채널을 0-255 범위로 정규화
        ch_min, ch_max = image_3ch[i].min(), image_3ch[i].max()
        if ch_max > ch_min:
            image_3ch_vis[i] = ((image_3ch[i] - ch_min) / (ch_max - ch_min) * 255).astype(np.uint8)
        else:
            image_3ch_vis[i] = np.zeros((256, 256), dtype=np.uint8)
    
    # PIL로 저장
    pil_image = Image.fromarray(image_3ch_vis.transpose(1, 2, 0))  # [H, W, C] 순서로 변환
    pil_image.save(output_path)
    print(f"3채널 입력 이미지 저장됨: {output_path}")
    
    return output_path

def create_comparison_visualization(s1_image: np.ndarray, predicted_mask: np.ndarray, 
                                   output_path: str, s2_image: np.ndarray = None) -> str:
    """S1, S2(선택), 예측 결과를 나란히 보여주는 비교 이미지 생성
    
    Args:
        s1_image: S1 입력 이미지 [H, W, 3] (RGB)
        predicted_mask: 예측된 마스크 [H, W]
        output_path: 출력 파일 경로
        s2_image: S2 입력 이미지 [H, W, 3] (RGB, 선택 사항)
    
    Returns:
        저장된 파일 경로
    """
    from PIL import Image, ImageDraw, ImageFont
    
    # 클래스별 컬러 매핑
    class_colors = {
        0: [0, 0, 0],        # Background
        1: [34, 139, 34],    # Forest
        2: [107, 142, 35],   # Shrubland
        3: [154, 205, 50],   # Grassland
        4: [0, 191, 255],    # Wetland
        5: [255, 165, 0],    # Cropland
        6: [128, 128, 128],  # Urban/Built-up
        7: [139, 69, 19],    # Barren
        8: [0, 0, 255],      # Water
    }
    
    # 마스크를 컬러 이미지로 변환
    h, w = predicted_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in class_colors.items():
        color_mask[predicted_mask == class_id] = color
    
    # 이미지 개수에 따라 레이아웃 결정
    num_images = 3 if s2_image is not None else 2
    
    # 각 이미지 크기
    img_h, img_w = 256, 256
    padding = 10
    title_height = 30
    
    # 전체 캔버스 크기
    canvas_w = num_images * (img_w + padding) + padding
    canvas_h = img_h + title_height + padding * 2
    
    # 캔버스 생성
    canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # 폰트 설정 (기본 폰트 사용)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # S1 이미지 추가
    x_offset = padding
    y_offset = title_height + padding
    
    # S1
    s1_pil = Image.fromarray(s1_image)
    canvas.paste(s1_pil, (x_offset, y_offset))
    draw.text((x_offset + img_w // 2 - 20, padding), "S1 Input", fill=(0, 0, 0), font=font)
    x_offset += img_w + padding
    
    # S2 (있다면)
    if s2_image is not None:
        s2_pil = Image.fromarray(s2_image)
        canvas.paste(s2_pil, (x_offset, y_offset))
        draw.text((x_offset + img_w // 2 - 20, padding), "S2 Input", fill=(0, 0, 0), font=font)
        x_offset += img_w + padding
    
    # 예측 결과
    pred_pil = Image.fromarray(color_mask)
    canvas.paste(pred_pil, (x_offset, y_offset))
    draw.text((x_offset + img_w // 2 - 30, padding), "Prediction", fill=(0, 0, 0), font=font)
    
    # 저장
    canvas.save(output_path)
    print(f"비교 이미지 저장됨: {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Segmentation 결과 시각화')
    parser.add_argument('input_mask', help='입력 마스크 파일 경로 (PNG 또는 TIF)')
    parser.add_argument('-o', '--output', help='출력 컬러 마스크 파일 경로 (기본: input_mask_vis.tif)')
    parser.add_argument('--format', choices=['png', 'tif'], default='tif', help='출력 형식 (기본: tif)')
    
    args = parser.parse_args()
    
    # 입력 파일 확인
    input_path = Path(args.input_mask)
    if not input_path.exists():
        print(f"오류: 입력 파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)
    
    # 출력 파일 경로 설정
    if args.output:
        output_path = args.output
    else:
        output_path = input_path.parent / f"{input_path.stem}_vis.{args.format}"
    
    try:
        # 마스크 로드
        if input_path.suffix.lower() == '.png':
            mask = load_mask_from_png(str(input_path))
        elif input_path.suffix.lower() in ['.tif', '.tiff']:
            mask = load_mask_from_tif(str(input_path))
        else:
            print(f"오류: 지원하지 않는 파일 형식: {input_path.suffix}")
            sys.exit(1)
        
        print(f"마스크 로드됨: {input_path}")
        print(f"마스크 데이터 타입: {mask.dtype}")
        print(f"마스크 값 범위: {mask.min()} ~ {mask.max()}")
        
        # 컬러 마스크 생성
        create_color_mask(mask, str(output_path), args.format)
        
    except Exception as e:
        print(f"오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
