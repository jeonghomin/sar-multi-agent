
import torch
from torch import nn
from typing import List
import sys
import os

from mmseg.models.necks import Feature2Pyramid
from mmseg.models.decode_heads import UPerHead, FCNHead
from app.utils.misc import resize

class SegmentationDecoder(nn.Module):
    """
    Segmentation Decoder - MMSegmentation UPerHead 사용 (Copernicus-FM과 동일)
    """
    def __init__(self, 
                 embed_dim: int = 768, 
                 num_classes: int = 19,
                 channels: int = 512,
                 dropout_ratio: float = 0.1,
                 pretrained_path: str = None):
        super().__init__()
        
      
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
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
            ),
        )
        
       
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)
    
    def _load_pretrained_weights(self, pretrained_path: str):
        """Pretrained weights 로드"""
        try:
            weights_dir = "/home/mjh/Project/LLM/RAG/ai-service/weights"
            full_path = os.path.join(weights_dir, pretrained_path)
            
            if not os.path.exists(full_path):
                print(f"Decoder weights 파일을 찾을 수 없습니다: {full_path}")
                return
            
            checkpoint = torch.load(full_path, map_location='cpu')
            
    
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
               
                state_dict = {k.replace('model.', '') if k.startswith('model.') else k: v for k, v in state_dict.items()}
            else:
                state_dict = checkpoint
            
           
            self.load_state_dict(state_dict, strict=False)
            
        except Exception as e:
            print(f"Segmentation Decoder weights 로드 실패: {e}")
        
    def forward(self, encoder_features: List[torch.Tensor], target_size: tuple = None) -> torch.Tensor:
        """
        Args:
            encoder_features: List of feature maps from encoder [B, C, H, W]
            target_size: Target output size (H, W)
        Returns:
            segmentation_logits: [B, num_classes, H, W]
        """
        
        feats = self.neck(encoder_features)
        out = self.decoder(feats)
        if target_size is not None:
            out = resize(out, size=target_size, mode="bilinear", align_corners=False)
        return out

