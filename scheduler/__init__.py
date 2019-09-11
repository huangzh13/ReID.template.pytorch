"""
-------------------------------------------------
   File Name:    __init__.py.py
   Author:       Zhonghao Huang
   Date:         2019/9/10
   Description:
-------------------------------------------------
"""

from torch.optim import lr_scheduler


def make_scheduler(cfg, optimizer):
    if cfg.SCHEDULER.NAME == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.SCHEDULER.STEP, gamma=cfg.SCHEDULER.GAMMA)
    else:
        raise KeyError("Unknown scheduler: ", cfg.SCHEDULER.NAME)

    return scheduler
