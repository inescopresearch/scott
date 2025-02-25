from pathlib import Path
import yaml
import torch
import random, os
import numpy as np

def get_params_from_yaml_file(fpath):
    params = None
    with open(Path(fpath), 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    return params

def seed_everything(seed=47):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_device(device_number, verbose=True):
    if torch.cuda.is_available():
        device = f'cuda:{int(device_number)}'
        assert device_number < torch.cuda.device_count(), f'Error, device {device} not available.'
    else:
        device = 'cpu'
    if verbose: print(f"Training on device: '{device}' - {torch.cuda.get_device_name(device_number)}")
    return device

def compute_gradient_norm(parameters):
    total_norm = 0.
    for p in parameters:
        if p.grad is not None and p.requires_grad:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm 