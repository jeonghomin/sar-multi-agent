# Vision Transformer Encoder - Copernicus-FM ViT 사용
import torch
import torch.nn as nn
import math
import sys
import os
from typing import Tuple, List

# Copernicus-FM ViT 모델 import를 위한 경로 추가
copernicus_path = "/home/mjh/Copernicus-FM/Copernicus-Bench/src"
if copernicus_path not in sys.path:
    sys.path.append(copernicus_path)

try:
    from foundation_models.ViT.vit import vit_base as vit_base_cls
    from foundation_models.ViT.vit import vit_small as vit_small_cls
    from foundation_models.ViT.vit import vit_large as vit_large_cls
    from foundation_models.ViT.vit_seg import vit_base as vit_base_seg
    from foundation_models.ViT.vit_seg import vit_small as vit_small_seg
    from foundation_models.ViT.vit_seg import vit_large as vit_large_seg
    COPERNICUS_VIT_AVAILABLE = True
except ImportError as e:
    print(f"Copernicus-FM ViT 모델을 불러올 수 없습니다: {e}")
    COPERNICUS_VIT_AVAILABLE = False

class VisionTransformerEncoder(nn.Module):
    """
    Foundation Encoder - ViT 기반
    """
    def __init__(self, 
                 img_size: int = 224, 
                 patch_size: int = 16, 
                 in_chans: int = 3,
                 embed_dim: int = 768, 
                 depth: int = 12, 
                 num_heads: int = 12, 
                 mlp_ratio: float = 4.0,
                 out_indices: List[int] = [3, 5, 7, 11]):
        super().__init__()
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, 0.0, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, drop_path=dpr[i])
            for i in range(depth)
        ])
        
        self.out_indices = out_indices
        self.embed_dim = embed_dim
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Extract features at different layers
        out_features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                # Remove cls token and reshape for decoder
                out = x[:, 1:]  # Remove cls token
                _, hw, D = out.shape
                H_shape = W_shape = int(math.sqrt(hw))
                out = out.reshape(B, H_shape, W_shape, D).permute(0, 3, 1, 2).contiguous()
                out_features.append(out)
        
        # Global features for classification
        global_features = x[:, 0]  # cls token
        
        return global_features, out_features

class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Attention(nn.Module):
    """Multi-head Attention"""
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class Mlp(nn.Module):
    """MLP"""
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class CopernicusViTEncoder(nn.Module):
    """
    Copernicus-FM ViT Encoder 래퍼
    """
    def __init__(self, 
                 vit_size: str = "base",
                 img_size: int = 224, 
                 patch_size: int = 16, 
                 in_chans: int = 3,
                 embed_dim: int = 768, 
                 depth: int = 12, 
                 num_heads: int = 12, 
                 mlp_ratio: float = 4.0,
                 out_indices: List[int] = [3, 5, 7, 11]):
        super().__init__()
        
        if not COPERNICUS_VIT_AVAILABLE:
            raise ImportError("Copernicus-FM ViT 모델을 사용할 수 없습니다.")
        
        self.vit_size = vit_size
        self.embed_dim = embed_dim
        self.out_indices = out_indices
        
        # Classification용 ViT (global features 추출용)
        if vit_size == "base":
            self.vit_cls = vit_base_cls(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                num_classes=0  # head 제거
            )
            self.embed_dim = 768
        elif vit_size == "small":
            self.vit_cls = vit_small_cls(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                num_classes=0
            )
            self.embed_dim = 384
        elif vit_size == "large":
            self.vit_cls = vit_large_cls(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                num_classes=0
            )
            self.embed_dim = 1024
        else:
            raise ValueError(f"지원하지 않는 ViT 크기: {vit_size}")
        
        # Segmentation용 ViT (multi-scale features 추출용)
        if vit_size == "base":
            self.vit_seg = vit_base_seg(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans
            )
        elif vit_size == "small":
            self.vit_seg = vit_small_seg(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans
            )
        elif vit_size == "large":
            self.vit_seg = vit_large_seg(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans
            )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            global_features: Global features for classification [B, embed_dim]
            multi_scale_features: Multi-scale features for detection/segmentation [List[Tensor]]
        """
        # Global features for classification
        global_features = self.vit_cls.forward_features(x)  # [B, embed_dim]
        
        # Multi-scale features for detection/segmentation
        multi_scale_features = self.vit_seg.forward_features(x)  # List of [B, C, H, W]
        
        return global_features, multi_scale_features
