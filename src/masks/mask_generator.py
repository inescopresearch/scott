import torch
from src.masks.random_uniform import get_random_uniform_mask
from src.masks.random_blockwise import get_random_blockwise_mask

class MaskGenerator:

    types2funcs = {
        'uniform': get_random_uniform_mask,
        'blockwise': get_random_blockwise_mask
    }

    def __init__(self, img_size, patch_size, masking_type):
        fmap_size = img_size // patch_size
        if not isinstance(fmap_size, tuple):
            fmap_size = (fmap_size, ) * 2
        self.shape = fmap_size
        self.height, self.width = fmap_size
        self.num_patches = self.height * self.width
        self.masking_function = self.types2funcs[masking_type]
    
    def get_shape(self):
        return self.height, self.width
    
    def __call__(self, batch_size, mask_ratio):
        """
        0 = mask patch, 1 = keep patch.
        Args:
        - batch_size: size of the batch.
        """
        target_masks = []
        for _ in range(batch_size):
            mask = self.masking_function(mask_shape=self.shape, mask_ratio=mask_ratio)
            target_masks.append(mask.flatten())
        target_masks = torch.stack(target_masks, dim=0)
        return target_masks