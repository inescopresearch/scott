import torch.nn as nn
import torch
import torch.nn.functional as F

class SimpleAttentiveClassifier(nn.Module):
    
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.patch_importance = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=embed_dim, out_features=1)
        )
        self.attn_dropout = nn.Dropout(0.2)
        self.global_2_logits = nn.Linear(in_features=embed_dim, out_features=num_classes)

    def forward(self, x):
        importances = self.patch_importance(x).squeeze(-1)
        global_importance = F.softmax(importances, dim=-1)
        if self.training:
            global_importance = self.attn_dropout(global_importance)
        global_repr = torch.einsum('b n i, b n k -> b i k', global_importance.unsqueeze(-1), x).squeeze(1)
        logits = self.global_2_logits(global_repr)
        return logits
