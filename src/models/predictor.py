import math
import torch.nn as nn
from src.models.transformer import Transformer

class Predictor(nn.Module):

    def __init__(self, 
                 embed_dim,
                 num_patches,
                 num_register_tokens,
                 backbone_depth,
                 depth, 
                 num_heads,
                 ffn_layer,
                 mlp_ratio,
                 dropout_rate,
                 attention_dropout,
                 stochastic_depth_rate):
        
        super().__init__()
        self.embed_dim = embed_dim
        self.backbone_depth = backbone_depth
        self.depth = depth
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.stochastic_depth_rate = stochastic_depth_rate
        self.attention_dropout = attention_dropout
        self.mlp_ratio = mlp_ratio

        self.transformer = Transformer(embed_dim=embed_dim,
                                       num_patches=num_patches,
                                       depth=self.depth,
                                       num_heads=self.num_heads,
                                       mlp_ratio=self.mlp_ratio,
                                       dropout_rate=self.dropout_rate,
                                       attention_dropout=self.attention_dropout,
                                       stochastic_depth_rate=self.stochastic_depth_rate,
                                       num_register_tokens=num_register_tokens,
                                       ffn_layer=ffn_layer)
        self.head = nn.Sequential(
            nn.SiLU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=True))
        
        self.fix_init_weight()

    def forward(self, x):
        x = self.transformer(x)
        x = self.head(x)
        return x

    def get_last_selfattention(self, x):
        return self.transformer.get_last_selfattention(x)
    
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        
        for block_id, block in enumerate(self.transformer.blocks):
            block_id += self.backbone_depth
            rescale(block.self_attn.proj.weight.data, block_id + 1)
            rescale(block.ffn_layer.last_linear.weight.data, block_id + 1)