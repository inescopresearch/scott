import torch.nn as nn
from einops import rearrange
from src.models.transformer import Transformer

class PatchEmbed(nn.Module):

    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)
    """
    def __init__(self, 
                 img_size,
                 patch_size,
                 in_channels,
                 embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.fmap_hw = img_size // patch_size
        self.num_patches = int(self.fmap_hw ** 2)
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.apply(self.init_weight)

    def forward(self, x):
        x = self.proj(x) # B C H W
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x
    
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

class ViT(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        embed_dim,
        in_channels,
        depth,
        num_heads,
        mlp_ratio,
        dropout_rate,
        attention_dropout,
        stochastic_depth_rate,
        num_register_tokens,
        ffn_layer
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.stochastic_depth_rate = stochastic_depth_rate
        self.num_register_tokens = num_register_tokens
        self.ffn_layer = ffn_layer

        self.tokenizer = PatchEmbed(img_size=img_size,
                                    patch_size=patch_size,
                                    in_channels=in_channels,
                                    embed_dim=embed_dim)
        self.num_patches = self.tokenizer.num_patches
        self.output_sequence_length = self.num_patches 
        
        self.transformer = Transformer(embed_dim=embed_dim,
                                       num_patches=self.tokenizer.num_patches,
                                       depth=self.depth,
                                       num_heads=self.num_heads,
                                       mlp_ratio=self.mlp_ratio,
                                       dropout_rate=self.dropout_rate,
                                       attention_dropout=self.attention_dropout,
                                       stochastic_depth_rate=self.stochastic_depth_rate,
                                       num_register_tokens=num_register_tokens,
                                       ffn_layer=ffn_layer)
        
    def forward(self, x, masks=None):
        x = self.tokenizer(x)
        x = self.transformer(x, masks)
        return x
    
    def get_last_selfattention(self, x, masks=None):
        x = self.tokenizer(x)
        attn = self.transformer.get_last_selfattention(x, masks)
        return attn
