import torch
from src.schedulers import get_warmup_flat_cosine_scheduler, LinearScheduler

class ScheduledOptimizer:

    def __init__(self, optimizer, lr_scheduler, wd_scheduler):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.wd_scheduler = wd_scheduler
        self.reset(num_step=0)

    def step(self):
        self.optimizer.step()
        # update lr value
        new_lr = self.lr_scheduler.step()
        self._update_lr(value=new_lr)
        # update wd value
        new_wd = self.wd_scheduler.step()
        self._update_wd(value=new_wd)

    def _update_lr(self, value):
        for group in self.optimizer.param_groups:
            group['lr'] = value

    def _update_wd(self, value):
        for group in self.optimizer.param_groups:
            if ('WD_exclude' not in group) or not group['WD_exclude']:
                group['weight_decay'] = value 

    def reset(self, num_step=0):
        new_lr = self.lr_scheduler.reset(num_step=num_step)
        self._update_lr(value=new_lr)
        new_wd = self.wd_scheduler.reset(num_step=num_step)        
        self._update_wd(value=new_wd)
    
    def zero_grad(self):
        self.optimizer.zero_grad()

    def info(self):
        for group in self.optimizer.param_groups:
            lr = group['lr']
            if ('WD_exclude' not in group) or not group['WD_exclude']:
                wd = group['weight_decay']

        info = {
            'lr': lr,
            'wd': wd
        }
        return info
    
def get_optimizer(model,
                  total_steps,
                  lr_warmup_steps,
                  lr_start,
                  lr_peak,
                  lr_final,
                  lr_flat_pctg,
                  wd_start,
                  wd_final):
    lr_scheduler = get_warmup_flat_cosine_scheduler(total_steps=total_steps,
                                                    warmup_steps=lr_warmup_steps,
                                                    start_value=lr_start,
                                                    peak_value=lr_peak,
                                                    final_value=lr_final,
                                                    flat_pctg=lr_flat_pctg)
    
    wd_scheduler = LinearScheduler(start_value=wd_start,
                                   final_value=wd_final,
                                   total_steps=total_steps)
    param_groups = [
        {
            'params': (p for n, p in model.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        },  {
            'params': (p for n, p in model.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]
    adamW_optim = torch.optim.AdamW(param_groups)
    optimizer = ScheduledOptimizer(optimizer=adamW_optim,
                                   lr_scheduler=lr_scheduler,
                                   wd_scheduler=wd_scheduler)
    return optimizer
