import torchvision.transforms as T
import numpy as np

IMAGENET_NORMALIZATION_STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

class TransformsPipeline(object):

    def __init__(self, 
                 crop_size, 
                 crop_scale, 
                 tfms_type, 
                 num_views = 1):
        assert tfms_type in ['none', 'same', 'diff'], 'Invalid tfms_type parameter. Should be one of: "none", "same", or "diff".'
        self.tfms_type = tfms_type
        assert num_views > 0, 'At least should generate 1 view.'
        self.num_views = num_views
        self.pre_tfms = T.Compose([
            T.RandomResizedCrop(size=crop_size, scale=crop_scale, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
        ])
        self.color_tfms = np.array([
            T.RandomApply(transforms=[T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            T.RandomGrayscale(p=.1),
            T.RandomApply(transforms=[T.GaussianBlur(kernel_size=9, sigma=(0.1, 2))], p=0.3),
        ])
        self.post_tfms = T.Compose([
            T.ToTensor(),
            T.Normalize(IMAGENET_NORMALIZATION_STATS[0], IMAGENET_NORMALIZATION_STATS[1])
        ])

    def __call__(self, x):
        x = self.pre_tfms(x)
        views = []
        for _ in range(self.num_views):
            if self.tfms_type != 'none' and np.random.uniform() > 0.4:
                color_tfms = T.Compose(self.color_tfms[np.random.permutation(len(self.color_tfms))])
                x_ = color_tfms(x)
            else:
                x_ = x
            view = self.post_tfms(x_)
            views.append(view)
        return views if len(views) > 1 else views[0]
    
def get_train_transforms(crop_size,
                         crop_scale,
                         tfms_type,
                         num_views):
    tfms = TransformsPipeline(crop_size=crop_size,
                              crop_scale=crop_scale,
                              tfms_type=tfms_type,
                              num_views=num_views)
    return tfms

def get_test_transforms(crop_size):
    tfms = T.Compose([
        T.Resize(256),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_NORMALIZATION_STATS[0], IMAGENET_NORMALIZATION_STATS[1])])
    return tfms