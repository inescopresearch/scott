from src.models.transformer import SwiGLUFFN

import math
import torch
import torch.nn as nn
from einops import rearrange


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attention_dropout, projection_dropout):
        super().__init__()
        self.heads = num_heads
        head_dim = embed_dim // self.heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, query, key_value):
        q = self.q(query)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        kv = self.kv(key_value).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)

        q = q * self.scale

        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')

        return self.proj_drop(self.proj(x)), attn

class CrossAttentionBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, attention_dropout, projection_dropout):
        super().__init__()
        self.pre_norm = nn.LayerNorm(normalized_shape=embed_dim)
        self.cross_attn = CrossAttention(embed_dim=embed_dim,
                                         num_heads=num_heads, 
                                         attention_dropout=attention_dropout, 
                                         projection_dropout=projection_dropout)
        self.norm = nn.LayerNorm(normalized_shape=embed_dim)
        self.ffn_layer = SwiGLUFFN(d_model=embed_dim)
    
    def forward(self, query, key_value, return_attention=False):
        y, attn = self.cross_attn(query=query, key_value=self.pre_norm(key_value))
        if return_attention:
            return attn
        query = query + y
        query = self.norm(query)
        query = query + self.ffn_layer(query)
        return query

class AttentivePooler(nn.Module):

    def __init__(self, embed_dim, num_heads, attention_dropout, projection_dropout):
        super().__init__()
        num_queries = 1
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        nn.init.trunc_normal_(self.query_tokens.data, std=.02)
        self.cross_attn_block = CrossAttentionBlock(embed_dim=embed_dim,
                                                    num_heads=num_heads,
                                                    attention_dropout=attention_dropout,
                                                    projection_dropout=projection_dropout)
    def forward(self, x):
        q = self.query_tokens.repeat(len(x), 1, 1)
        q = self.cross_attn_block(query=q, key_value=x)
        return q
        
class CrossAttentiveClassifier(nn.Module):
    """Attentive Classifier Head
    https://arxiv.org/pdf/2202.03026 5.2. explanation...
    """

    def __init__(self, 
                 embed_dim,
                 num_classes, 
                 num_heads, 
                 attention_dropout,
                 projection_dropout):
                 
        super().__init__()
        self.attentive_pooler = AttentivePooler(embed_dim=embed_dim,
                                                num_heads=num_heads,
                                                attention_dropout=attention_dropout,
                                                projection_dropout=projection_dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, num_classes, bias=True)

        self.apply(self.init_weight) 
    
    def forward(self, x):
        x = self.attentive_pooler(x).squeeze(1)
        x = self.norm(x)
        x = self.linear(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and (m.bias is not None):
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

