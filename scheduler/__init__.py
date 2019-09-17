"""
-------------------------------------------------
   File Name:    __init__.py.py
   Author:       Zhonghao Huang
   Date:         2019/9/10
   Description:
-------------------------------------------------
"""

from torch.optim import lr_scheduler
from scheduler.warmup_scheduler import GradualWarmupScheduler


def make_scheduler(cfg, optimizer):
    if cfg.SCHEDULER.NAME == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.SCHEDULER.STEP, gamma=cfg.SCHEDULER.GAMMA)
    elif cfg.SCHEDULER.NAME == 'WarmupStepLR':
        step_scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.SCHEDULER.STEP, gamma=cfg.SCHEDULER.GAMMA)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=cfg.SCHEDULER.WARMUP_FACTOR,
                                           total_epoch=cfg.SCHEDULER.WARMUP_ITERS, after_scheduler=step_scheduler)
        pass
    else:
        raise KeyError("Unknown scheduler: ", cfg.SCHEDULER.NAME)

    return scheduler
