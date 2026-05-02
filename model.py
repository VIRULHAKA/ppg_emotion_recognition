import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

class ppgEncoder(nn.Module):  # CLIP style encoder for PPG signals
    def __init__(self, in_channels=1, emb_dim=128):
        super().__init__()
    
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(128, emb_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x).squeeze(-1)  # (B, 128)
        x = self.projection(x)        # (B, emb_dim)
        x = F.normalize(x, dim=-1)    # L2 normalize
        return x
    
class gsrEncoder(nn.Module):  # CLIP style encoder for GSR signals
    def __init__(self, in_channels=1, emb_dim=128):
        super().__init__()
    
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(128, emb_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x).squeeze(-1)  # (B, 128)
        x = self.projection(x)        # (B, emb_dim)
        x = F.normalize(x, dim=-1)    # L2 normalize
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        return self.attn(x, x, x)[0]


class ppgTransformerEncoder(nn.Module): # ViT style Transformer encoder for PPG signals
    def __init__(self, embed_dim=128, num_heads=8, mlp_dim=256):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 400 + 1, embed_dim))  
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), 
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        batch_size = x.size(0)
        cls_t = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_t, x], dim=1)  
        x = x + self.pos_embed
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        cls_output = x[:, 0, :]  # (B, D)
        cls_output = F.normalize(cls_output, dim=-1) 
        return cls_output
    

class gsrTransformerEncoder(nn.Module): # ViT style Transformer encoder for GSR signals
    def __init__(self, embed_dim=128, num_heads=8, mlp_dim=256):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 400 + 1, embed_dim))  
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), 
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        batch_size = x.size(0)
        cls_t = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_t, x], dim=1)  
        x = x + self.pos_embed
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        cls_output = x[:, 0, :]  # (B, D)
        cls_output = F.normalize(cls_output, dim=-1) 
        return cls_output
    

class fusionModule(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim * 2, emb_dim)
        self.fc2 = nn.Linear(emb_dim, emb_dim)
        self.GELU = nn.GELU()

    def forward(self, z1, z2):
        # z1, z2: (B, emb_dim)
        combined = torch.cat([z1, z2], dim=-1)  # (B, emb_dim*2)
        fused = self.fc1(combined)                # (B, emb_dim)
        fused = self.GELU(fused)
        fused = self.fc2(fused)                  # (B, emb_dim)
        return fused
    

class emotionClassifier(nn.Module):
    def __init__(self, emb_dim=128, num_classes=5):
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, z):
        # z: (B, emb_dim)
        return self.fc(z)  # (B, num_classes)
    


