import random
import numpy as np
import torch

def get_random_uniform_mask(mask_shape, mask_ratio):
    num_patches = np.prod(mask_shape)
    num_masked_patches = (1-mask_ratio) * num_patches
    if num_masked_patches - int(num_masked_patches) > 0:
       num_masked_patches = int(num_masked_patches) + 1

    mask = np.zeros(num_patches)
    mask[random.sample(range(0, num_patches), num_masked_patches)] = 1
    assert mask.sum() == num_masked_patches, f"mask: {mask}, mask count {mask.sum()}"
    
    return torch.tensor(mask)