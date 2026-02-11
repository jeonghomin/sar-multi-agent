# Classification Head
import torch
from torch import nn

class ClassificationDecoder(nn.Module):
    """
    Classification Decoder
    """
    def __init__(self, embed_dim: int = 768, num_classes: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.dropout(x)
        x = self.head(x)
        return x
