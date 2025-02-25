# Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
# Copyright Zhun Zhong & Liang Zheng
# 
# Hacked together by / Copyright 2020 Ross Wightman
# 
# Modified by Hangbo Bao, for generating the masked position for visual image transformer
# 
# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
# Copyright Zhun Zhong & Liang Zheng
#
# Hacked together by / Copyright 2020 Ross Wightman
#
# Modified by Hangbo Bao, for generating the masked position for visual image transformer
# --------------------------------------------------------'
# Adapted by Carlos Velez Garcia for MIM-JEPA and SCOTT integration.

import math
import random
import numpy as np
import torch

log_aspect_ratio = (math.log(0.3), math.log(3.3))

def _mask(mask, max_mask_patches):
    height, width = mask.shape
    delta = 0
    for _ in range(10):
        target_area = random.uniform(16, max_mask_patches)
        aspect_ratio = math.exp(random.uniform(*log_aspect_ratio))
        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))
        if w < width and h < height:
            top = random.randint(0, height - h)
            left = random.randint(0, width - w)
            num_masked = mask[top: top + h, left: left + w].sum()
            # Overlap
            if 0 < h * w - num_masked <= max_mask_patches:
                for i in range(top, top + h):
                    for j in range(left, left + w):
                        if mask[i, j] == 0:
                            mask[i, j] = 1
                            delta += 1
            if delta > 0:
                break
    return mask

def get_random_blockwise_mask(mask_shape, mask_ratio):
    num_patches = np.prod(mask_shape)
    num_masked_patches = (1-mask_ratio) * num_patches
    if num_masked_patches - int(num_masked_patches) > 0:
       num_masked_patches = int(num_masked_patches) + 1
    mask = np.zeros(shape=mask_shape, dtype=np.int32)
    pending_masked_patches = num_masked_patches - mask.sum()
    while pending_masked_patches > 0:
        mask = _mask(mask=mask, max_mask_patches=pending_masked_patches)
        pending_masked_patches = num_masked_patches - mask.sum()
    assert mask.sum() == num_masked_patches, f"mask: {mask}, mask count {mask.sum()}"
    return torch.tensor(mask)
