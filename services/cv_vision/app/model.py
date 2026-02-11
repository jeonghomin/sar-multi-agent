# PyTorch 모델 레퍼 (로드/ 추론) - Backward Compatibility
# 새로운 모델은 app.model 패키지에서 import
from .model import (
    VisionTransformerEncoder,
    ClassificationDecoder,
    ObjectDetectionDecoder, 
    SegmentationDecoder,
    MultiTaskModel,
    MultiTaskWrapper
)
from .model import MultiTaskWrapper as DualTaskWrapper
from .model import MultiTaskModel as DualTaskModel

import os
from typing import Optional, Tuple, List
import torch
from torch import nn
from PIL import Image
import numpy as np
import math
from functools import partial

class Segmodel(nn.Module):
    """
    기존 간단한 모델 (호환성을 위해 유지)
    """
    def __init__(self):
        super().__init__()
        
    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _,_, H, W = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1,1, H, device=x.device),
            torch.linspace(-1,1,W,device=x.device),
            indexing = "ij"
        )

        r = torch.sqrt(xx**2 + yy**2)
        mask = ( 1 - r ).clamp(0,1)

        return mask.unsqueeze(0).unsqueeze(0)

class SegmodelWrapper:
    """
    기존 호환성을 위한 wrapper (deprecated)
    """
    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Segmodel().to(self.device).eval()

    @torch.inference_mode()
    def predict_mask(self, image_path: str, threshold: float = 0.5) -> np.ndarray:
        pil = Image.open(image_path).convert("RGB")
        W, H = pil.size

        arr = np.array(pil).astype(np.float32) / 255.0
        ten = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(self.device)

        prob = self.model(ten)
        prob = prob.squeeze().clamp(0,1).cpu().numpy()

        mask = (prob >= threshold).astype(np.uint8) * 255
        return mask

# 사용 예제
if __name__ == "__main__":
    # 모델 초기화
    wrapper = DualTaskWrapper(
        img_size=224,
        num_classes_cls=1000,  # ImageNet 클래스 수
        num_classes_det=80,    # COCO 클래스 수
        num_queries=100
    )
    
    # 예제 이미지 경로 (실제 이미지로 교체 필요)
    image_path = "example_image.jpg"
    
    try:
        # Classification 수행
        cls_id, cls_confidence = wrapper.predict_classification(image_path)
        print(f"Classification 결과: 클래스 {cls_id}, 신뢰도 {cls_confidence:.3f}")
        
        # Object Detection 수행
        detections = wrapper.predict_detection(image_path, confidence_threshold=0.5)
        print(f"Detection 결과: {len(detections)}개 객체 검출")
        for i, det in enumerate(detections):
            print(f"  객체 {i+1}: 클래스 {det['class_id']}, 신뢰도 {det['confidence']:.3f}, bbox {det['bbox']}")
            
    except FileNotFoundError:
        print("예제 이미지 파일을 찾을 수 없습니다. 실제 이미지 경로로 교체해주세요.")