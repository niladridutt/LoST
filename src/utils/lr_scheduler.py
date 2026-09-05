from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler

def build_scheduler(optimizer, n_epoch, n_iter_per_epoch, lr_min=0, warmup_steps=0, warmup_lr_init=0, decay_steps=None, cosine_lr=True):
    if decay_steps is None:
        decay_steps = n_epoch * n_iter_per_epoch
    
    if cosine_lr:
        scheduler = CosineLRScheduler(optimizer, t_initial=decay_steps, lr_min=lr_min, warmup_t=warmup_steps, warmup_lr_init=warmup_lr_init, 
                                      cycle_limit=1, t_in_epochs=False, warmup_prefix=True)
    else:
        scheduler = StepLRScheduler(optimizer, decay_t=decay_steps, warmup_t=warmup_steps, warmup_lr_init=warmup_lr_init, 
                                    t_in_epochs=False, warmup_prefix=True)
    
    return scheduler
