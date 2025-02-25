import torch
from torch import nn, einsum
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange, repeat

def get_2d_sincos_pos_embed(embed_dim, fmap_size_hw, num_register_tokens=0):
    """
    Mostly copy-pasted from: https://github.com/facebookresearch/jepa/blob/main/src/models/utils/pos_embs.py

    grid_size: int of the grid height and width
    returns:
        pos_embed: [fmap_size_hw*fmap_size_hw, embed_dim] (w/o num_register_tokens)
                or [1+fmap_size_hw*fmap_size_hw, embed_dim] (w/ num_register_tokens)
    """
    grid_h = np.arange(fmap_size_hw, dtype=float)
    grid_w = np.arange(fmap_size_hw, dtype=float)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)  # order of meshgrid is very important for indexing as [h, w]

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h)  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w)  # (H*W, D/2)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    if num_register_tokens > 0:
        pos_embed = np.concatenate([np.zeros([num_register_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    returns: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def closest_multiple_of_x_to_n(x, n):
    return int(x * np.round(n / x))

class SwiGLUFFN(nn.Module):
    """
    https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/swiglu_ffn.py
    """
    def __init__(self, d_model):
        super().__init__()
        dim_feedforward = closest_multiple_of_x_to_n(x=128, n=(8/3*d_model))
        self.w12 = nn.Linear(in_features=d_model, out_features=2 * dim_feedforward, bias=False)
        self.last_linear = nn.Linear(in_features=dim_feedforward, out_features=d_model, bias=False)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.last_linear(hidden)
        
class MlpFFN(nn.Module):

    def __init__(self, d_model, dim_feedforward, dropout):
        super().__init__()
        self.linear1  = nn.Linear(in_features=d_model, out_features=dim_feedforward, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.last_linear  = nn.Linear(dim_feedforward, d_model, bias=False)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.last_linear(x)
        x = self.dropout2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.heads = num_heads
        head_dim = dim // self.heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        q = q * self.scale

        attn = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = einsum('b h i j, b h j d -> b h i d', attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')

        return self.proj_drop(self.proj(x)), attn
    
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        batch, drop_prob, device, dtype = x.shape[0], self.drop_prob, x.device, x.dtype

        if drop_prob <= 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (batch, *((1,) * (x.ndim - 1)))

        keep_mask = torch.zeros(shape, device = device).float().uniform_(0, 1) < keep_prob
        output = x.div(keep_prob) * keep_mask.float()
        return output

class Block(nn.Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and rwightman's timm package.
    """
    def __init__(self, d_model, nhead, dropout,
                 attention_dropout, drop_path_rate, ffn_layer):
        super().__init__()

        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)
        self.drop_path = DropPath(drop_path_rate)
        self.ffn_layer = ffn_layer

    def forward(self, x, return_attention=False, *args, **kwargs):
        y, attn = self.self_attn(F.layer_norm(x, (x.size(-1),)))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = F.layer_norm(x, (x.size(-1),))
        x = x + self.drop_path(self.ffn_layer(x))
        return x

class Transformer(nn.Module):

    def __init__(self, 
                 embed_dim, 
                 num_patches,
                 depth, 
                 num_heads,
                 mlp_ratio,
                 dropout_rate,
                 attention_dropout,
                 stochastic_depth_rate,
                 num_register_tokens,
                 ffn_layer):
        
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.num_patches = num_patches 
        self.depth = depth
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.stochastic_depth_rate = stochastic_depth_rate
        self.attention_dropout = attention_dropout
        self.num_register_tokens = num_register_tokens

        def get_ffn_layer(ffn_layer):
            if ffn_layer == 'swiglu':
                return SwiGLUFFN(d_model=self.embed_dim)
            elif ffn_layer == 'mlp':
                return MlpFFN(d_model=self.embed_dim, 
                              dim_feedforward=int(self.embed_dim * self.mlp_ratio),
                              dropout=self.dropout_rate)
            
        self.total_sequence_length = self.num_register_tokens + self.num_patches
        # Registers embedding
        if self.num_register_tokens > 0:
            self.reg_embed = nn.Parameter(torch.zeros(1, self.num_register_tokens, self.embed_dim), requires_grad=True)
            nn.init.trunc_normal_(self.reg_embed.data, std=.02)
        # Mask embedding
        self.msk_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim), requires_grad=True)
        nn.init.trunc_normal_(self.msk_embed, std=1e-6)

        # Position embedding
        self.pos_embed = nn.Parameter(torch.FloatTensor(get_2d_sincos_pos_embed(embed_dim=self.embed_dim,
                                                                                fmap_size_hw=int(math.sqrt(self.num_patches)),
                                                                                num_register_tokens=self.num_register_tokens)), 
                                      requires_grad=False)        
        # Transformer blocks
        self.position_dropout = nn.Dropout(p=self.dropout_rate)
        dpr = [x.item() for x in torch.linspace(0, self.stochastic_depth_rate, self.depth)]

        self.blocks = nn.ModuleList(
            [Block(d_model=self.embed_dim,
                   nhead=self.num_heads,
                   dropout=self.dropout_rate,
                   attention_dropout=self.attention_dropout,
                   drop_path_rate=layer_dpr,
                   ffn_layer=get_ffn_layer(ffn_layer)) 
            for layer_dpr in dpr]
        )
        
        self.apply(self.init_weight) 
        self.fix_init_weight()

    def prepare_tokens(self, x, masks=None):
        b = x.shape[0]
        # Apply mask to tokens.
        if masks is not None:
            x[~masks.bool()] = self.msk_embed # Replace tokens where masks == False.
        # Append register tokens
        if self.num_register_tokens > 0:
            reg_tokens = repeat(self.reg_embed, '1 n d -> b n d', b = b)
            x = torch.cat((reg_tokens, x), dim=1)
        # Add position embedding
        x = x + self.pos_embed

        x = self.position_dropout(x)
        return x

    def forward(self, x, masks=None):
        x = self.prepare_tokens(x=x, masks=masks)
        for blk in self.blocks:
            x = blk(x)
        # Drop register tokens
        x = x[:, self.num_register_tokens:]
        return x
    
    def get_last_selfattention(self, x, masks=None):
        x = self.prepare_tokens(x=x, masks=masks)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and (m.bias is not None):
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        
        for block_id, block in enumerate(self.blocks):
            rescale(block.self_attn.proj.weight.data, block_id + 1)
            rescale(block.ffn_layer.last_linear.weight.data, block_id + 1)


