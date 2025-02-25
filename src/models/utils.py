
import torch

def compute_num_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model_checkpoint(root_path, model, tag='model', verbose=True):
    path = root_path/f'{tag}_checkpoint.pth'
    torch.save(model.state_dict(), f=path)
    if verbose: print(f'Saved model checkpoint at: {path}')
    return path

def load_model_checkpoint(file_path, model, device):
    try:
        model.load_state_dict(torch.load(file_path, map_location=device, weights_only=True))
    except Exception as e:
        print(f'Encountered exception when loading checkpoint {e}')
    return model