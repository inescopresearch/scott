import src.models.sparsify as sparsify
from src.models.transformer import Transformer
from src.models.layers.max_blur_pool import MaxBlurPool2D

from einops import rearrange
import torch.nn as nn
import torch

class SparseCNNTokenizer(nn.Module):
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
        act_layer = nn.ReLU
        
        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            act_layer(),
            MaxBlurPool2D(kernel_size=3, stride=2, max_pool_size=2, ceil_mode=False),
            nn.Conv2d(in_channels=64, out_channels=embed_dim, kernel_size=7, stride=2, padding=3, bias=False),
            act_layer(),
            MaxBlurPool2D(kernel_size=3, stride=2, max_pool_size=2, ceil_mode=False),
        ]

        self.tokenizer = sparsify.dense_model_to_sparse(nn.Sequential(*layers))
        self.apply(self.init_weight)

    def _apply_masks(self, x, masks=None):
        if masks is None:
            fmap_mask = torch.ones((x.shape[0], 1, self.fmap_hw, self.fmap_hw), dtype=torch.bool).to(x.device)
        else:
            B, SEQ_LENGTH = masks.shape
            assert B == x.shape[0], f'Invalid batch shapes, expected one mask for each input x.'
            assert SEQ_LENGTH == self.num_patches, f'Invalid sequence length, expected {self.num_patches} found {masks.shape[1]}'
            fmap_mask = masks.view(B, self.fmap_hw, self.fmap_hw).unsqueeze(1).to(x.device).bool()
        sparsify._cur_active = fmap_mask
        mask = fmap_mask.repeat_interleave(self.patch_size, 2).repeat_interleave(self.patch_size, 3)  # (B, 1, H, W)
        x = x * mask
        return x
    
    def forward(self, x, masks=None):
        x = self._apply_masks(x, masks)
        x = self.tokenizer(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

# Sparse Convolutional Tokenizer Transformer Model
class SCOTT(nn.Module):
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

        self.tokenizer = SparseCNNTokenizer(img_size=img_size,
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
        x = self.tokenizer(x, masks)
        x = self.transformer(x, masks)
        return x
    
    def get_last_selfattention(self, x, masks=None):
        x = self.tokenizer(x, masks)
        attn = self.transformer.get_last_selfattention(x, masks)
        return attn
