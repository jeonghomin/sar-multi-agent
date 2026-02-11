# Integrated Multi-Task Model
import torch
from torch import nn
from typing import Tuple, List, Optional
import os
import numpy as np
from PIL import Image

from .encoder import VisionTransformerEncoder, CopernicusViTEncoder
from .copernicus_encoder import CopernicusFMEncoder, CopernicusFMMultiTaskModel
from .classification_head import ClassificationDecoder
from .detection_head import ObjectDetectionDecoder
from .segmentation_head import SegmentationDecoder

class MultiTaskModel(nn.Module):
    """
    통합 Multi-Task 모델 - Encoder + 3개 Head
    """
    def __init__(self, 
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 num_classes_cls: int = 1000,
                 num_classes_det: int = 80,
                 num_classes_seg: int = 19,
                 num_queries: int = 100,
                 use_simple_seg: bool = True,
                 use_copernicus_vit: bool = True,
                 vit_size: str = "base",
                 use_copernicus_fm: bool = False,
                 pretrained_path: Optional[str] = None,
                 language_embed: Optional[str] = None,
                 key: str = "S2"):
        super().__init__()
        
        # Foundation Encoder 선택
        if use_copernicus_fm:
            try:
                self.encoder = CopernicusFMEncoder(
                    model_size=vit_size,
                    img_size=img_size,
                    patch_size=patch_size,
                    in_chans=in_chans,
                    pretrained_path=pretrained_path,
                    language_embed=language_embed,
                    key=key
                )
                # Copernicus-FM의 실제 embed_dim 사용
                embed_dim = self.encoder.embed_dim
                print(f"Copernicus-FM Encoder 사용: {vit_size}, embed_dim={embed_dim}")
            except ImportError as e:
                print(f"Copernicus-FM 사용 실패, Copernicus ViT 사용: {e}")
                if use_copernicus_vit:
                    try:
                        self.encoder = CopernicusViTEncoder(
                            vit_size=vit_size,
                            img_size=img_size,
                            patch_size=patch_size,
                            in_chans=in_chans,
                            embed_dim=embed_dim
                        )
                        # Copernicus ViT의 실제 embed_dim 사용
                        embed_dim = self.encoder.embed_dim
                        print(f"Copernicus ViT 사용: {vit_size}, embed_dim={embed_dim}")
                    except ImportError as e2:
                        print(f"Copernicus ViT 사용 실패, 기본 ViT 사용: {e2}")
                        self.encoder = VisionTransformerEncoder(
                            img_size=img_size,
                            patch_size=patch_size,
                            in_chans=in_chans,
                            embed_dim=embed_dim,
                            depth=depth,
                            num_heads=num_heads
                        )
                else:
                    self.encoder = VisionTransformerEncoder(
                        img_size=img_size,
                        patch_size=patch_size,
                        in_chans=in_chans,
                        embed_dim=embed_dim,
                        depth=depth,
                        num_heads=num_heads
                    )
        elif use_copernicus_vit:
            try:
                self.encoder = CopernicusViTEncoder(
                    vit_size=vit_size,
                    img_size=img_size,
                    patch_size=patch_size,
                    in_chans=in_chans,
                    embed_dim=embed_dim
                )
                # Copernicus ViT의 실제 embed_dim 사용
                embed_dim = self.encoder.embed_dim
                print(f"Copernicus-FM ViT 사용: {vit_size}, embed_dim={embed_dim}")
            except ImportError as e:
                print(f"Copernicus ViT 사용 실패, 기본 ViT 사용: {e}")
                self.encoder = VisionTransformerEncoder(
                    img_size=img_size,
                    patch_size=patch_size,
                    in_chans=in_chans,
                    embed_dim=embed_dim,
                    depth=depth,
                    num_heads=num_heads
                )
        else:
            self.encoder = VisionTransformerEncoder(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads
            )
        
        # Classification Head
        self.classification_head = ClassificationDecoder(
            embed_dim=embed_dim,
            num_classes=num_classes_cls
        )
        
        # Object Detection Head
        self.detection_head = ObjectDetectionDecoder(
            embed_dim=embed_dim,
            num_classes=num_classes_det,
            num_queries=num_queries,
            num_heads=num_heads
        )
        
        # Segmentation Head (UPerNet 사용)
       
        self.segmentation_head = SegmentationDecoder(
            embed_dim=embed_dim,
            num_classes=num_classes_seg,
            channels=512,
            dropout_ratio=0.1
        )
        
    def forward(self, x: torch.Tensor, task: str = "all") -> dict:
        """
        Args:
            x: Input tensor [B, C, H, W]
            task: "classification", "detection", "segmentation", or "all"
        Returns:
            Dictionary with task-specific outputs
        """
        # Encoder forward
        global_features, multi_scale_features = self.encoder(x)
        
        outputs = {}
        
        if task in ["classification", "all"]:
            # Classification
            cls_logits = self.classification_head(global_features)
            outputs["classification"] = cls_logits
        
        if task in ["detection", "all"]:
            # Object Detection
            det_class_logits, det_bbox_coords = self.detection_head(multi_scale_features)
            outputs["detection"] = {
                "class_logits": det_class_logits,
                "bbox_coords": det_bbox_coords
            }
        
        if task in ["segmentation", "all"]:
            # Segmentation
            if hasattr(self.segmentation_head, 'forward') and len(multi_scale_features) > 0:
                seg_logits = self.segmentation_head(multi_scale_features)
            else:
                # Simple segmentation (기존 방식)
                seg_logits = self.segmentation_head(x)
            outputs["segmentation"] = seg_logits
        
        return outputs

class MultiTaskWrapper:
    """
    Multi-Task Model Wrapper
    """
    def __init__(self, 
                 weights_path: Optional[str] = None, 
                 device: Optional[str] = None,
                 img_size: int = 224,
                 num_classes_cls: int = 1000,
                 num_classes_det: int = 80,
                 num_classes_seg: int = 19,
                 num_queries: int = 100,
                 use_simple_seg: bool = True,
                 use_copernicus_vit: bool = True,
                 vit_size: str = "base",
                 use_copernicus_fm: bool = False,
                 pretrained_path: Optional[str] = None,
                 language_embed: Optional[str] = None,
                 key: str = "S2"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        
        # Initialize model
        self.model = MultiTaskModel(
            img_size=img_size,
            num_classes_cls=num_classes_cls,
            num_classes_det=num_classes_det,
            num_classes_seg=num_classes_seg,
            num_queries=num_queries,
            use_simple_seg=use_simple_seg,
            use_copernicus_vit=use_copernicus_vit,
            vit_size=vit_size,
            use_copernicus_fm=use_copernicus_fm,
            pretrained_path=pretrained_path,
            language_embed=language_embed,
            key=key
        ).to(self.device).eval()
        
        # Load weights if provided
        if weights_path and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print(f"모델 가중치 로드됨: {weights_path}")
    
    @torch.inference_mode()
    def predict_classification(self, image_path: str) -> Tuple[int, float]:
        """Classification 수행"""
        # 이미지 로드 및 전처리
        pil = Image.open(image_path).convert("RGB")
        pil = pil.resize((self.img_size, self.img_size))
        
        # 텐서로 변환
        arr = np.array(pil).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # 모델 추론
        outputs = self.model(tensor, task="classification")
        cls_logits = outputs["classification"]
        
        # Softmax 적용
        cls_probs = torch.softmax(cls_logits, dim=1).squeeze(0).cpu().numpy()
        
        predicted_class = np.argmax(cls_probs)
        confidence = cls_probs[predicted_class]
        
        return predicted_class, confidence
    
    @torch.inference_mode()
    def predict_detection(self, image_path: str, confidence_threshold: float = 0.5) -> List[dict]:
        """Object Detection 수행"""
        # 이미지 로드 및 전처리
        pil = Image.open(image_path).convert("RGB")
        pil = pil.resize((self.img_size, self.img_size))
        
        # 텐서로 변환
        arr = np.array(pil).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # 모델 추론
        outputs = self.model(tensor, task="detection")
        det_outputs = outputs["detection"]
        
        det_class_logits = det_outputs["class_logits"].squeeze(0).cpu().numpy()
        det_bbox_coords = det_outputs["bbox_coords"].squeeze(0).cpu().numpy()
        
        # Softmax 적용
        det_class_probs = torch.softmax(torch.from_numpy(det_class_logits), dim=-1).numpy()
        
        detections = []
        for i in range(det_class_probs.shape[0]):
            class_probs = det_class_probs[i]
            max_class = np.argmax(class_probs)
            max_confidence = class_probs[max_class]
            
            if max_confidence > confidence_threshold and max_class < det_class_probs.shape[1] - 1:
                detection = {
                    'class_id': max_class,
                    'confidence': max_confidence,
                    'bbox': det_bbox_coords[i].tolist()
                }
                detections.append(detection)
        
        return detections
    
    @torch.inference_mode()
    def predict_segmentation(self, image_path: str, threshold: float = 0.5) -> np.ndarray:
        """Segmentation 수행"""
        # 이미지 로드 및 전처리
        pil = Image.open(image_path).convert("RGB")
        pil = pil.resize((self.img_size, self.img_size))
        
        # 텐서로 변환
        arr = np.array(pil).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # 모델 추론
        outputs = self.model(tensor, task="segmentation")
        seg_logits = outputs["segmentation"]
        
        # 마스크 생성
        prob = seg_logits.squeeze().clamp(0, 1).cpu().numpy()
        mask = (prob >= threshold).astype(np.uint8) * 255
        
        return mask
    
    @torch.inference_mode()
    def predict_all(self, image_path: str) -> dict:
        """모든 작업 수행"""
        # 이미지 로드 및 전처리
        pil = Image.open(image_path).convert("RGB")
        pil = pil.resize((self.img_size, self.img_size))
        
        # 텐서로 변환
        arr = np.array(pil).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # 모델 추론
        outputs = self.model(tensor, task="all")
        
        # 결과 처리
        results = {}
        
        # Classification
        if "classification" in outputs:
            cls_logits = outputs["classification"]
            cls_probs = torch.softmax(cls_logits, dim=1).squeeze(0).cpu().numpy()
            predicted_class = np.argmax(cls_probs)
            confidence = cls_probs[predicted_class]
            results["classification"] = {
                "class_id": predicted_class,
                "confidence": confidence
            }
        
        # Detection
        if "detection" in outputs:
            det_outputs = outputs["detection"]
            det_class_logits = det_outputs["class_logits"].squeeze(0).cpu().numpy()
            det_bbox_coords = det_outputs["bbox_coords"].squeeze(0).cpu().numpy()
            results["detection"] = {
                "class_logits": det_class_logits,
                "bbox_coords": det_bbox_coords
            }
        
        # Segmentation
        if "segmentation" in outputs:
            seg_logits = outputs["segmentation"]
            prob = seg_logits.squeeze().clamp(0, 1).cpu().numpy()
            results["segmentation"] = prob
        
        return results
