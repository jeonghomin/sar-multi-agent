# UPerNet Decoder - Copernicus-FM 스타일
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any
import sys
import os

# MMSegmentation import를 위한 경로 추가
try:
    # MMSegmentation이 설치되어 있다면 사용
    from mmseg.models.necks import Feature2Pyramid
    from mmseg.models.decode_heads import UPerHead, FCNHead
    from util.misc import resize
    MMSEG_AVAILABLE = True
except ImportError:
    print("MMSegmentation을 사용할 수 없습니다. 간단한 UPerNet 구현을 사용합니다.")
    MMSEG_AVAILABLE = False

class SimpleFeature2Pyramid(nn.Module):
    """간단한 Feature Pyramid Network 구현"""
    def __init__(self, embed_dim: int = 768, rescales: List[float] = [4, 2, 1, 0.5]):
        super().__init__()
        self.embed_dim = embed_dim
        self.rescales = rescales
        
        # 각 스케일별 conv layer
        self.convs = nn.ModuleList()
        for i, scale in enumerate(rescales):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
                    nn.BatchNorm2d(embed_dim),
                    nn.ReLU(inplace=True)
                )
            )
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: List of feature maps from encoder [B, C, H, W]
        Returns:
            pyramid_features: List of processed features
        """
        pyramid_features = []
        
        for i, (feat, conv) in enumerate(zip(features, self.convs)):
            # Feature processing
            processed_feat = conv(feat)
            pyramid_features.append(processed_feat)
        
        return pyramid_features

class SimpleUPerHead(nn.Module):
    """간단한 UPerHead 구현"""
    def __init__(self, 
                 in_channels: List[int],
                 in_index: List[int] = [0, 1, 2, 3],
                 pool_scales: tuple = (1, 2, 3, 6),
                 channels: int = 512,
                 dropout_ratio: float = 0.1,
                 num_classes: int = 19,
                 align_corners: bool = False):
        super().__init__()
        
        self.in_channels = in_channels
        self.in_index = in_index
        self.pool_scales = pool_scales
        self.channels = channels
        self.num_classes = num_classes
        self.align_corners = align_corners
        
        # PSP Module
        self.psp_modules = nn.ModuleList()
        for pool_scale in pool_scales:
            self.psp_modules.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    nn.Conv2d(in_channels[0], channels // len(pool_scales), 1),
                    nn.BatchNorm2d(channels // len(pool_scales)),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Feature fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels + in_channels[0], channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_ratio),
            nn.Conv2d(channels, num_classes, 1)
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of feature maps [B, C, H, W]
        Returns:
            segmentation_logits: [B, num_classes, H, W]
        """
        # Use the highest resolution feature
        x = features[0]  # [B, C, H, W]
        B, C, H, W = x.shape
        
        # PSP Module
        psp_outs = []
        for psp_module in self.psp_modules:
            psp_out = psp_module(x)
            psp_out = F.interpolate(psp_out, size=(H, W), mode='bilinear', align_corners=self.align_corners)
            psp_outs.append(psp_out)
        
        # Concatenate PSP outputs
        psp_cat = torch.cat(psp_outs, dim=1)  # [B, channels, H, W]
        
        # Fuse with original feature
        x_fused = torch.cat([x, psp_cat], dim=1)  # [B, C+channels, H, W]
        x_fused = self.fusion_conv(x_fused)  # [B, channels, H, W]
        
        # Classification
        segmentation_logits = self.classification_head(x_fused)  # [B, num_classes, H, W]
        
        return segmentation_logits

class SimpleFCNHead(nn.Module):
    """간단한 FCNHead 구현"""
    def __init__(self, 
                 in_channels: int,
                 in_index: int = 2,
                 channels: int = 256,
                 num_convs: int = 1,
                 concat_input: bool = False,
                 dropout_ratio: float = 0.1,
                 num_classes: int = 19,
                 align_corners: bool = False):
        super().__init__()
        
        self.in_channels = in_channels
        self.in_index = in_index
        self.channels = channels
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.num_classes = num_classes
        self.align_corners = align_corners
        
        # Convolution layers
        convs = []
        for i in range(num_convs):
            in_ch = in_channels if i == 0 else channels
            convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, channels, 3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                )
            )
        self.convs = nn.Sequential(*convs)
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        
        # Classification head
        self.classification_head = nn.Conv2d(channels, num_classes, 1)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of feature maps [B, C, H, W]
        Returns:
            segmentation_logits: [B, num_classes, H, W]
        """
        # Use specified feature
        x = features[self.in_index]  # [B, C, H, W]
        
        # Convolution layers
        x = self.convs(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Classification
        segmentation_logits = self.classification_head(x)
        
        return segmentation_logits

class UPerNetDecoder(nn.Module):
    """
    UPerNet Decoder - Copernicus-FM 스타일
    """
    def __init__(self, 
                 embed_dim: int = 768,
                 num_classes: int = 19,
                 channels: int = 512,
                 dropout_ratio: float = 0.1,
                 use_mmseg: bool = True):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.channels = channels
        
        if use_mmseg and MMSEG_AVAILABLE:
            # MMSegmentation 사용
            self.neck = Feature2Pyramid(embed_dim=embed_dim, rescales=[4, 2, 1, 0.5])
            self.decoder = UPerHead(
                in_channels=[embed_dim] * 4,
                in_index=[0, 1, 2, 3],
                pool_scales=(1, 2, 3, 6),
                channels=channels,
                dropout_ratio=dropout_ratio,
                num_classes=num_classes,
                norm_cfg=dict(type="SyncBN", requires_grad=True),
                align_corners=False,
            )
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
            )
        else:
            # 간단한 구현 사용
            self.neck = SimpleFeature2Pyramid(embed_dim=embed_dim, rescales=[4, 2, 1, 0.5])
            self.decoder = SimpleUPerHead(
                in_channels=[embed_dim] * 4,
                in_index=[0, 1, 2, 3],
                pool_scales=(1, 2, 3, 6),
                channels=channels,
                dropout_ratio=dropout_ratio,
                num_classes=num_classes,
                align_corners=False
            )
            self.aux_head = SimpleFCNHead(
                in_channels=embed_dim,
                in_index=2,
                channels=256,
                num_convs=1,
                concat_input=False,
                dropout_ratio=dropout_ratio,
                num_classes=num_classes,
                align_corners=False
            )
    
    def forward(self, features: List[torch.Tensor], target_size: tuple = None) -> tuple:
        """
        Args:
            features: List of feature maps from encoder [B, C, H, W]
            target_size: Target output size (H, W)
        Returns:
            main_output: Main segmentation output [B, num_classes, H, W]
            aux_output: Auxiliary segmentation output [B, num_classes, H, W]
        """
        # Feature pyramid processing
        pyramid_features = self.neck(features)
        
        # Main decoder
        main_output = self.decoder(pyramid_features)
        
        # Auxiliary decoder
        aux_output = self.aux_head(pyramid_features)
        
        # Resize to target size if provided
        if target_size is not None:
            main_output = F.interpolate(
                main_output, size=target_size, mode='bilinear', align_corners=False
            )
            aux_output = F.interpolate(
                aux_output, size=target_size, mode='bilinear', align_corners=False
            )
        
        return main_output, aux_output

def resize(input, size, mode='bilinear', align_corners=False):
    """Resize function for compatibility"""
    return F.interpolate(input, size=size, mode=mode, align_corners=align_corners)
