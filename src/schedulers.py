import math

class Scheduler(object):

    def __init__(self, start_value=None, final_value=None, total_steps=None):
        self.start_value = start_value
        self.final_value = final_value
        self.total_steps = total_steps
        self._step = 0.
        self.value = 0.

    def step(self):
        self._step += 1
        self.value = self._compute_value(current_step=self._step)
        return self.value
    
    def reset(self, num_step=0.):
        self._step = max(1, num_step)
        self.value = self._compute_value(current_step=self._step)
        return self.value
    
    def _compute_value(self, current_step):
        raise NotImplementedError
    
    def plot(self, title, ylabel):
        import matplotlib.pyplot as plt

        plt.plot([self._compute_value(s) for s in range(self.total_steps)])
        plt.title(title)
        plt.xlabel('Total steps')
        plt.ylabel(ylabel)
        plt.show()
    
    def info(self):
        return {
            'value': self.value,
            'step': self._step
        }

class ConstantScheduler(Scheduler):

    def __init__(self, value, total_steps):
        super().__init__(start_value=value, final_value=value, total_steps=total_steps)
        self.value = value
    
    def _compute_value(self, current_step):
        return self.value

class LinearScheduler(Scheduler):

    def __init__(self, start_value, final_value, total_steps):
        super().__init__(start_value=start_value, final_value=final_value, total_steps=total_steps)

    def _compute_value(self, current_step):
        progress = float(current_step) / float(max(1, self.total_steps))
        new_value = self.start_value + progress * (self.final_value - self.start_value)
        return new_value

class CosineScheduler(Scheduler):
    
    def __init__(self, start_value, final_value, total_steps):
        super().__init__(start_value=start_value, final_value=final_value, total_steps=total_steps)

    def _compute_value(self, current_step):
        progress = float(current_step) / float(max(1, self.total_steps))
        new_value = self.final_value + (self.start_value - self.final_value) * 0.5 * (1. + math.cos(math.pi * progress))
        if self.final_value <= self.start_value:
            new_value = max(self.final_value, new_value)
        else:
            new_value = min(self.final_value, new_value)
        return new_value
    
class FlatCosineScheduler(Scheduler):

    def __init__(self, start_value, final_value, total_steps, flat_pctg):
        super().__init__(start_value=start_value, final_value=final_value, total_steps=total_steps) 
        self.start_value = start_value
        self.final_value = final_value 
        self.total_steps = total_steps
        self.flat_steps = int(total_steps * flat_pctg)
        self.cosine_steps = total_steps - self.flat_steps
        self.cosine_scheduler = CosineScheduler(start_value=start_value, final_value=self.final_value, total_steps=self.cosine_steps)

        self._step = 0.

    def _compute_value(self, current_step):
        if current_step < self.flat_steps:
            new_value = self.start_value
        else:
            new_value = self.cosine_scheduler._compute_value(current_step=current_step - self.flat_steps)
        return new_value
    
class SlopedCosineScheduler(Scheduler):

    def __init__(self, start_value, final_value, total_steps, decay_pctg, peak_decay):
        super().__init__(start_value=start_value, final_value=final_value, total_steps=total_steps) 
        self.start_value = start_value
        self.final_value = final_value 
        self.total_steps = total_steps
        self.decay_steps = int(total_steps * decay_pctg)
        self.peak_decay_value = start_value * peak_decay
        self.sloped_scheduler  = LinearScheduler(start_value=start_value, final_value=self.peak_decay_value, total_steps=self.decay_steps)
        self.cosine_steps = total_steps - self.decay_steps
        self.cosine_scheduler = CosineScheduler(start_value=self.peak_decay_value, final_value=self.final_value, total_steps=self.cosine_steps)

        self._step = 0.

    def _compute_value(self, current_step):
        if current_step < self.decay_steps:
            new_value = self.sloped_scheduler._compute_value(current_step=current_step)
        else:
            new_value = self.cosine_scheduler._compute_value(current_step=current_step - self.decay_steps)
        return new_value
    
class WarmupScheduler(Scheduler):
    def __init__(self, warmup_steps, start_value, peak_value, scheduler, total_steps):
        super().__init__(start_value=start_value, final_value=scheduler.final_value, total_steps=total_steps)
        self.warmup_steps = warmup_steps
        self.start_value = start_value
        self.peak_value = peak_value
        self.warmup_scheduler = LinearScheduler(start_value=self.start_value, final_value=self.peak_value, total_steps=self.warmup_steps)
        self.scheduler = scheduler
        self._step = 0.

    def _compute_value(self, current_step):
        if current_step < self.warmup_steps:
            new_value = self.warmup_scheduler._compute_value(current_step=current_step)
        else:
            new_value = self.scheduler._compute_value(current_step=current_step - self.warmup_steps)
        return new_value

def get_warmup_cosine_scheduler(total_steps,
                                warmup_steps,  
                                start_value, 
                                peak_value,
                                final_value):
    scheduler_steps = total_steps - warmup_steps
    cosine_scheduler = CosineScheduler(start_value=peak_value,
                                       final_value=final_value,
                                       total_steps=scheduler_steps)
    
    warmup_scheduler = WarmupScheduler(warmup_steps=warmup_steps,
                                       start_value=start_value,
                                       peak_value=peak_value,
                                       scheduler=cosine_scheduler,
                                       total_steps=total_steps)
    return warmup_scheduler

def get_warmup_flat_cosine_scheduler(total_steps,
                                     warmup_steps,
                                     start_value,
                                     peak_value,
                                     final_value,
                                     flat_pctg
                                     ):
    scheduler_steps = total_steps - warmup_steps
    flat_cosine_scheduler = FlatCosineScheduler(start_value=peak_value,
                                                final_value=final_value,
                                                total_steps=scheduler_steps,
                                                flat_pctg=flat_pctg)
    warmup_scheduler = WarmupScheduler(warmup_steps=warmup_steps,
                                       start_value=start_value,
                                       peak_value=peak_value,
                                       scheduler=flat_cosine_scheduler,
                                       total_steps=total_steps)
    return warmup_scheduler    

def get_warmup_sloped_cosine_scheduler(total_steps,
                                       warmup_steps,
                                       start_value,
                                       peak_value,
                                       final_value,
                                       decay_pctg,
                                       peak_decay):
    scheduler_steps = total_steps - warmup_steps
    sloped_cosine_scheduler = SlopedCosineScheduler(start_value=peak_value,
                                                    final_value=final_value,
                                                    total_steps=scheduler_steps,
                                                    decay_pctg=decay_pctg,
                                                    peak_decay=peak_decay)    
    warmup_scheduler = WarmupScheduler(warmup_steps=warmup_steps,
                                       start_value=start_value,
                                       peak_value=peak_value,
                                       scheduler=sloped_cosine_scheduler,
                                       total_steps=total_steps)
    return warmup_scheduler   

if __name__ == '__main__':

    total_steps = 600

    scheduler = CosineScheduler(total_steps=total_steps, start_value=0.0001, final_value=0.000001)
    scheduler.reset()
    scheduler.plot(title='Cosine LR', ylabel='LR')

    scheduler = get_warmup_cosine_scheduler(warmup_steps=20, total_steps=total_steps, start_value=0.006, peak_value=0.01, final_value=0.000001)
    scheduler.reset()
    scheduler.plot(title='Warmup Cosine LR', ylabel='LR')

    scheduler = LinearScheduler(start_value=0.996, final_value=1.0, total_steps=total_steps)
    scheduler.reset()
    scheduler.plot(title='EMA Momentum Update', ylabel='EMA momentum')

    scheduler = get_warmup_flat_cosine_scheduler(warmup_steps=10, start_value=0.000001, peak_value=0.0005, final_value=0.00001, total_steps=total_steps, flat_pctg=0.8)
    scheduler.reset()
    scheduler.plot(title='Warmup Flat Cosine LR', ylabel='LR')


    scheduler = get_warmup_sloped_cosine_scheduler(warmup_steps=10, start_value=0.000001, peak_value=0.0005, final_value=0.00001, total_steps=total_steps, decay_pctg=0.8, peak_decay=0.7)
    scheduler.reset()
    scheduler.plot(title='Warmup Sloped Cosine LR', ylabel='LR')