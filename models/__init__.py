"""
-------------------------------------------------
   File Name:    __init__.py.py
   Author:       Zhonghao Huang
   Date:         2019/9/9
   Description:
-------------------------------------------------
"""

import torch.nn as nn

from models.baseline import Baseline, PCB
from models.losses import TripletLoss
from models.losses.smooth import CrossEntropyLabelSmooth

MODEL = {
    'baseline': Baseline,
    'PCB': PCB
}


def make_model(cfg, num_classes):
    if cfg.MODEL.NAME not in MODEL:
        raise KeyError("Unknown model: ", cfg.MODEL.NAME)
    else:
        model = MODEL[cfg.MODEL.NAME](num_classes, arch=cfg.MODEL.ARCH, stride=cfg.MODEL.STRIDE)

    return model


def make_loss(cfg, num_classes):
    if cfg.MODEL.LABEL_SMOOTH:
        xent_criterion = CrossEntropyLabelSmooth(num_classes=num_classes)
    else:
        xent_criterion = nn.CrossEntropyLoss()

    if cfg.SOLVER.LOSS == 'softmax_triplet':
        embedding_criterion = TripletLoss(margin=cfg.SOLVER.MARGIN)

        def criterion(softmax_y, triplet_y, labels):
            sum_loss = [embedding_criterion(output, labels)[0] for output in triplet_y] + \
                       [xent_criterion(output, labels) for output in softmax_y]
            loss = sum(sum_loss)
            return loss

        return criterion
    else:
        raise KeyError("Unknown loss: ", cfg.SOLVER.LOSS)
