import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange

class MultiHeadAttentiveClassifier(nn.Module):

    def __init__(self, embed_dim, num_classes, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes 
        self.num_heads = num_heads

        self.patch_importance = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=embed_dim, out_features=num_heads),
        )
        self.head_projectors = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=embed_dim, out_features=embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.global_2_logits = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=embed_dim, out_features=embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )

        self.importance_dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        h_emb = self.head_projectors(x)
        h_emb = rearrange(h_emb, 'b n (h d) -> b h n d', h = self.num_heads)

        h_imp = self.patch_importance(x).permute(0,2,1)
        h_imp = F.softmax(h_imp, dim=-1)
        if self.training:
            h_imp = self.importance_dropout(h_imp)
        h_imp = h_imp.unsqueeze(-1)

        h_global = torch.einsum('b h n i, b h n e -> b h i e', h_imp, h_emb)
        h_global = rearrange(h_global, 'b h i e -> b i (h e)')
        logits = self.global_2_logits(h_global).squeeze(1)
        return logits
