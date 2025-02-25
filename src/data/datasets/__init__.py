from torch.utils.data import DataLoader
import torch
import random
import numpy as np
from pathlib import Path
from os.path import expanduser

from src.data.datasets.flowers102 import get_flowers102, get_flowers102_full, get_flowers102_full_eval, get_flowers102_inv, get_flowers102_inv_eval
from src.data.datasets.imagenet100 import get_imagenet100, get_imagenet100_full, get_imagenet100_full_eval
from src.data.datasets.pets37 import get_pets37, get_pets37_full, get_pets37_full_eval

DATASETS = {
    'flowers102': get_flowers102,
    'flowers102_full': get_flowers102_full,
    'flowers102_full_eval': get_flowers102_full_eval,
    'flowers102_inv': get_flowers102_inv,
    'flowers102_inv_eval': get_flowers102_inv_eval,
    'imagenet100': get_imagenet100,
    'imagenet100_full': get_imagenet100_full,
    'imagenet100_full_eval': get_imagenet100_full_eval,
    'pets37': get_pets37,
    'pets37_full': get_pets37_full,
    'pets37_full_eval': get_pets37_full_eval
}

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataset(name, datasets_root, batch_size, num_workers, tfms_train, tfms_test, seed, verbose=True):
    assert name in DATASETS.keys(), f'Error invalid dataset. Choose one of: {DATASETS.keys()}'
    
    if datasets_root == '': 
        datasets_root = Path(expanduser('~'))/'datasets'
        print(f'Using default dataset folder location: {datasets_root}')

    ds_train, ds_test, ds_valid = DATASETS[name](root=Path(datasets_root), 
                                                 tfms_train=tfms_train,
                                                 tfms_test=tfms_test)
        
    g = torch.Generator()
    g.manual_seed(seed)

    dl_train = DataLoader(dataset=ds_train, 
                          batch_size=batch_size, 
                          shuffle=True,
                          drop_last=True,
                          num_workers=num_workers,
                          worker_init_fn=seed_worker,
                          generator=g)

    dl_test = DataLoader(dataset=ds_test,
                         batch_size=batch_size, 
                         shuffle=False,
                         num_workers=num_workers)
    
    if ds_valid is not None:
        dl_valid = DataLoader(dataset=ds_valid,
                              batch_size=batch_size, 
                              shuffle=False,
                              num_workers=num_workers)
    else:
        dl_valid = None

    if verbose:
        print(f'Loaded Dataset: {name}')
        print(f'- Train num samples: {len(dl_train.dataset)} - num classes: {len(dl_train.dataset.classes)}')
        if dl_valid is not None:
            print(f'- Valid num samples: {len(dl_valid.dataset)} - num classes: {len(dl_valid.dataset.classes)}')
        else:
            print(f'- No validation split available.')
        print(f'- Test  num samples: {len(dl_test.dataset)} - num classes: {len(dl_test.dataset.classes)}')

    return dl_train, dl_test, dl_valid