# Object Detection Head
import torch
from torch import nn
from typing import List, Tuple

class MLP(nn.Module):
    """Multi-Layer Perceptron for bbox regression"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x

class ObjectDetectionDecoder(nn.Module):
    """
    Object Detection Decoder - DETR 스타일
    """
    def __init__(self, 
                 embed_dim: int = 768, 
                 num_classes: int = 80, 
                 num_queries: int = 100,
                 num_heads: int = 8,
                 num_layers: int = 6):
        super().__init__()
        
        self.num_queries = num_queries
        self.num_classes = num_classes
        
        # Object queries (learnable embeddings)
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output heads
        self.class_embed = nn.Linear(embed_dim, num_classes + 1)  # +1 for background
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)  # 4 for bbox coordinates
        
    def forward(self, encoder_features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Use the highest resolution feature map
        x = encoder_features[-1]  # [B, C, H, W]
        B, C, H, W = x.shape
        
        # Flatten spatial dimensions
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Object queries
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, num_queries, C]
        
        # Decoder
        decoder_output = self.transformer_decoder(query_embed, x)
        
        # Output predictions
        class_logits = self.class_embed(decoder_output)  # [B, num_queries, num_classes+1]
        bbox_coords = self.bbox_embed(decoder_output).sigmoid()  # [B, num_queries, 4]
        
        return class_logits, bbox_coords
