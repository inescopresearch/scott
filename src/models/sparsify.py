# Mostly copy-pasted from https://github.com/keyu-tian/SparK
import torch
import torch.nn as nn
from src.models.layers.max_blur_pool import MaxBlurPool2D

_cur_active: torch.Tensor = None            # B1ff

def _get_active_ex_or_ii(H, W, returning_active_ex=True):
    h_repeat, w_repeat = H // _cur_active.shape[-2], W // _cur_active.shape[-1]
    active_ex = _cur_active.repeat_interleave(h_repeat, dim=2).repeat_interleave(w_repeat, dim=3)
    return active_ex if returning_active_ex else active_ex.squeeze(1).nonzero(as_tuple=True)  # ii: bi, hi, wi

def sp_conv_forward(self, x: torch.Tensor):
    x = super(type(self), self).forward(x)
    x *= _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=True)    # (BCHW) *= (B1HW), mask the output of conv
    return x

def sp_bn_forward(self, x: torch.Tensor):
    ii = _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=False)
    
    bhwc = x.permute(0, 2, 3, 1)
    nc = bhwc[ii]                               # select the features on non-masked positions to form a flatten feature `nc`
    nc = super(type(self), self).forward(nc)    # use BN1d to normalize this flatten feature `nc`
    
    bchw = torch.zeros_like(bhwc)
    bchw[ii] = nc
    bchw = bchw.permute(0, 3, 1, 2)
    return bchw

class SparseConv2d(nn.Conv2d):
    forward = sp_conv_forward   # hack: override the forward function; see `sp_conv_forward` above for more details

class SparseMaxPooling(nn.MaxPool2d):
    forward = sp_conv_forward   # hack: override the forward function; see `sp_conv_forward` above for more details

class SparseMaxBlurPool2d(MaxBlurPool2D):
    forward = sp_conv_forward # hack: override the forward function; see `sp_conv_forward` above for more details

class SparseAvgPooling(nn.AvgPool2d):
    forward = sp_conv_forward   # hack: override the forward function; see `sp_conv_forward` above for more details

class SparseBatchNorm2d(nn.BatchNorm1d):
    forward = sp_bn_forward     # hack: override the forward function; see `sp_bn_forward` above for more details

class SparseSyncBatchNorm2d(nn.SyncBatchNorm):
    forward = sp_bn_forward     # hack: override the forward function; see `sp_bn_forward` above for more details

class SparseLayerNorm(nn.LayerNorm):
    r""" 
    Sparse LayerNorm with shape (batch_size, channels, height, width).
    """
    
    def __init__(self, normalized_shape, eps=1e-6, sparse=True):
        super().__init__(normalized_shape, eps, elementwise_affine=True)
        self.sparse = sparse
    
    def forward(self, x):
        if x.ndim == 4: # BCHW
            if self.sparse:
                ii = _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=False)
                bhwc = x.permute(0, 2, 3, 1)
                nc = bhwc[ii]
                nc = super(SparseLayerNorm, self).forward(nc)
            
                x = torch.zeros_like(bhwc)
                x[ii] = nc
                return x.permute(0, 3, 1, 2)
            else:
                u = x.mean(1, keepdim=True)
                s = (x - u).pow(2).mean(1, keepdim=True)
                x = (x - u) / torch.sqrt(s + self.eps)
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
                return x
        else:           # BLC or BC
            if self.sparse:
                raise NotImplementedError
            else:
                return super(SparseLayerNorm, self).forward(x)

    def __repr__(self):
        return super(SparseLayerNorm, self).__repr__()[:-1] + f', ch=BCHW, sp={self.sparse})'

def dense_model_to_sparse(model: nn.Module, verbose=False, sbn=False):
    m = model
    oup = m
    if isinstance(m, nn.Conv2d):
        m: nn.Conv2d
        bias = m.bias is not None
        oup = SparseConv2d(
            m.in_channels, m.out_channels,
            kernel_size=m.kernel_size, stride=m.stride, padding=m.padding,
            dilation=m.dilation, groups=m.groups, bias=bias, padding_mode=m.padding_mode,
        )
        oup.weight.data.copy_(m.weight.data)
        if bias:
            oup.bias.data.copy_(m.bias.data)
    elif isinstance(m, nn.MaxPool2d):
        m: nn.MaxPool2d
        oup = SparseMaxPooling(m.kernel_size, stride=m.stride, padding=m.padding, dilation=m.dilation, return_indices=m.return_indices, ceil_mode=m.ceil_mode)
    elif isinstance(m, MaxBlurPool2D):
        m: MaxBlurPool2D
        oup = SparseMaxBlurPool2d(kernel_size=m.kernel_size, stride=m.stride, max_pool_size=m.max_pool_size, ceil_mode=m.ceil_mode)
    elif isinstance(m, nn.AvgPool2d):
        m: nn.AvgPool2d
        oup = SparseAvgPooling(m.kernel_size, m.stride, m.padding, ceil_mode=m.ceil_mode, count_include_pad=m.count_include_pad, divisor_override=m.divisor_override)
    elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
        m: nn.BatchNorm2d
        oup = (SparseSyncBatchNorm2d if sbn else SparseBatchNorm2d)(m.weight.shape[0], eps=m.eps, momentum=m.momentum, affine=m.affine, track_running_stats=m.track_running_stats)
        oup.weight.data.copy_(m.weight.data)
        oup.bias.data.copy_(m.bias.data)
        oup.running_mean.data.copy_(m.running_mean.data)
        oup.running_var.data.copy_(m.running_var.data)
        oup.num_batches_tracked.data.copy_(m.num_batches_tracked.data)
        if hasattr(m, "qconfig"):
            oup.qconfig = m.qconfig
    elif isinstance(m, nn.LayerNorm) and not isinstance(m, SparseLayerNorm):
        m: nn.LayerNorm
        oup = SparseLayerNorm(m.weight.shape[0], eps=m.eps)
        oup.weight.data.copy_(m.weight.data)
        oup.bias.data.copy_(m.bias.data)

    elif isinstance(m, (nn.Conv1d,)):
        raise NotImplementedError
    
    for name, child in m.named_children():
        oup.add_module(name, dense_model_to_sparse(child, verbose=verbose, sbn=sbn))
    del m
    return oup