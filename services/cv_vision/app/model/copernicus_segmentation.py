# Copernicus-FM Segmentation - MMSegmentation 사용 (원본과 동일)
import torch
import torch.nn as nn
from typing import List, Optional
import sys
import os

# MMSegmentation import를 위한 경로 추가
try:
    from mmseg.models.necks import Feature2Pyramid
    from mmseg.models.decode_heads import UPerHead, FCNHead
    from util.misc import resize
    MMSEG_AVAILABLE = True
except ImportError:
    print("MMSegmentation을 사용할 수 없습니다.")
    MMSEG_AVAILABLE = False

class CopernicusFMSegmentation(nn.Module):
    """
    Copernicus-FM Segmentation - 원본과 동일한 구조
    """
    def __init__(self, 
                 embed_dim: int = 768,
                 num_classes: int = 19,
                 channels: int = 512,
                 dropout_ratio: float = 0.1,
                 ignore_index: int = 255):
        super().__init__()
        
        if not MMSEG_AVAILABLE:
            raise ImportError("MMSegmentation이 필요합니다.")
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        # Feature Pyramid Network (Copernicus-FM과 동일)
        self.neck = Feature2Pyramid(embed_dim=embed_dim, rescales=[4, 2, 1, 0.5])
        
        # UPerHead (Copernicus-FM과 동일)
        self.decoder = UPerHead(
            in_channels=[embed_dim] * 4,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=channels,
            dropout_ratio=dropout_ratio,
            num_classes=num_classes,
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
            ),
        )
        
        # FCNHead (Auxiliary head, Copernicus-FM과 동일)
        self.aux_head = FCNHead(
            in_channels=embed_dim,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=dropout_ratio,
            num_classes=num_classes,
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4
            ),
        )
        
        # Loss function (Copernicus-FM과 동일)
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, encoder_features: List[torch.Tensor], target_size: tuple = None) -> tuple:
        """
        Args:
            encoder_features: List of feature maps from encoder [B, C, H, W]
            target_size: Target output size (H, W)
        Returns:
            main_output: Main segmentation output [B, num_classes, H, W]
            aux_output: Auxiliary segmentation output [B, num_classes, H, W]
        """
        # Feature pyramid processing (Copernicus-FM과 동일)
        feats = self.neck(encoder_features)
        
        # Main decoder
        out = self.decoder(feats)
        if target_size is not None:
            out = resize(out, size=target_size, mode="bilinear", align_corners=False)
        
        # Auxiliary decoder
        out_a = self.aux_head(feats)
        if target_size is not None:
            out_a = resize(out_a, size=target_size, mode="bilinear", align_corners=False)
        
        return out, out_a
    
    def loss(self, outputs, labels):
        """Loss function (Copernicus-FM과 동일)"""
        return self.criterion(outputs[0], labels) + 0.4 * self.criterion(outputs[1], labels)
    
    def params_to_optimize(self):
        """Optimizable parameters (Copernicus-FM과 동일)"""
        return (
            list(self.neck.parameters())
            + list(self.decoder.parameters())
            + list(self.aux_head.parameters())
        )

class CopernicusFMSegmentationWrapper:
    """
    Copernicus-FM Segmentation Wrapper for easy inference
    """
    def __init__(self, 
                 embed_dim: int = 768,
                 num_classes: int = 19,
                 channels: int = 512,
                 dropout_ratio: float = 0.1,
                 ignore_index: int = 255,
                 device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = CopernicusFMSegmentation(
            embed_dim=embed_dim,
            num_classes=num_classes,
            channels=channels,
            dropout_ratio=dropout_ratio,
            ignore_index=ignore_index
        ).to(self.device).eval()
    
    @torch.inference_mode()
    def predict(self, encoder_features: List[torch.Tensor], target_size: tuple = None) -> torch.Tensor:
        """
        Args:
            encoder_features: List of feature maps from encoder [B, C, H, W]
            target_size: Target output size (H, W)
        Returns:
            segmentation_logits: [B, num_classes, H, W]
        """
        # Move features to device
        encoder_features = [feat.to(self.device) for feat in encoder_features]
        
        # Forward pass
        main_output, aux_output = self.model(encoder_features, target_size)
        
        # Return main output only (auxiliary is for training)
        return main_output
    
    @torch.inference_mode()
    def predict_with_aux(self, encoder_features: List[torch.Tensor], target_size: tuple = None) -> tuple:
        """
        Args:
            encoder_features: List of feature maps from encoder [B, C, H, W]
            target_size: Target output size (H, W)
        Returns:
            main_output: Main segmentation output [B, num_classes, H, W]
            aux_output: Auxiliary segmentation output [B, num_classes, H, W]
        """
        # Move features to device
        encoder_features = [feat.to(self.device) for feat in encoder_features]
        
        # Forward pass
        main_output, aux_output = self.model(encoder_features, target_size)
        
        return main_output, aux_output
