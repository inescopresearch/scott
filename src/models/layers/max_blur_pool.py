"""
Mostly copy-pasted from: https://github.com/kornia/kornia
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxBlurPool2D(nn.Module):
    r"""Compute pools and blurs and downsample a given feature map.

    See :cite:`zhang2019shiftinvar` for more details.

    Args:
        kernel_size: the kernel size for max pooling.
        stride: stride for pooling.
        max_pool_size: the kernel size for max pooling.
        ceil_mode: should be true to match output size of conv2d with same kernel size.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H / stride, W / stride)`

    Returns:
        torch.Tensor: the transformed tensor.
    """
    def __init__(self, kernel_size, stride, max_pool_size, ceil_mode):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.max_pool_size = max_pool_size
        self.ceil_mode = ceil_mode
        self.kernel = get_pascal_kernel_2d(kernel_size, norm=True)

    def forward(self, input):
        self.kernel = torch.as_tensor(self.kernel, device=input.device, dtype=input.dtype)
        return _max_blur_pool_by_kernel2d(
            input, self.kernel.repeat((input.size(1), 1, 1, 1)), self.stride, self.max_pool_size, self.ceil_mode
        )

def get_pascal_kernel_2d(kernel_size, norm):
    """Generate pascal filter kernel by kernel size.

    Args:
        kernel_size: height and width of the kernel.
        norm: if to normalize the kernel or not. Default: True.

    Returns:
        if kernel_size is an integer the kernel will be shaped as :math:`(kernel_size, kernel_size)`
        otherwise the kernel will be shaped as :math: `kernel_size`
    """
    ky, kx = _unpack_2d_ks(kernel_size)
    ax = get_pascal_kernel_1d(kx)
    ay = get_pascal_kernel_1d(ky)

    filt = ay[:, None] * ax[None, :]
    if norm:
        filt = filt / torch.sum(filt)
    return filt

def _unpack_2d_ks(kernel_size):
    if isinstance(kernel_size, int):
        ky = kx = kernel_size
    else:
        assert len(kernel_size) == 2, "2D Kernel size should have a length of 2."
        ky, kx = kernel_size
    ky = int(ky)
    kx = int(kx)
    return (ky, kx)

def get_pascal_kernel_1d(kernel_size, norm=False):
    """Generate Yang Hui triangle (Pascal's triangle) by a given number.

    Args:
        kernel_size: height and width of the kernel.
        norm: if to normalize the kernel or not. Default: False.

    Returns:
        kernel shaped as :math:`(kernel_size,)`
    """
    pre: list[float] = []
    cur: list[float] = []
    for i in range(kernel_size):
        cur = [1.0] * (i + 1)

        for j in range(1, i // 2 + 1):
            value = pre[j - 1] + pre[j]
            cur[j] = value
            if i != 2 * j:
                cur[-j - 1] = value
        pre = cur

    out = torch.tensor(cur)

    if norm:
        out = out / out.sum()

    return out

def _max_blur_pool_by_kernel2d(input, kernel, stride, max_pool_size, ceil_mode):
    """Compute max_blur_pool by a given :math:`CxC_(out, None)xNxN` kernel."""
    assert len(kernel.shape) == 4 and kernel.shape[-2] == kernel.shape[-1], f"Invalid kernel shape. Expect CxC_outxNxN, Got {kernel.shape}"
    # compute local maxima
    input = F.max_pool2d(input, kernel_size=max_pool_size, padding=0, stride=1, ceil_mode=ceil_mode)
    # blur and downsample
    padding = _compute_zero_padding((kernel.shape[-2], kernel.shape[-1]))
    return F.conv2d(input, kernel, padding=padding, stride=stride, groups=input.size(1))

def _compute_zero_padding(kernel_size):
    r"""Utility function that computes zero padding tuple."""
    ky, kx = _unpack_2d_ks(kernel_size)
    return (ky - 1) // 2, (kx - 1) // 2