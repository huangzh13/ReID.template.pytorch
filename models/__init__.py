"""
-------------------------------------------------
   File Name:    __init__.py.py
   Author:       Zhonghao Huang
   Date:         2019/9/9
   Description:
-------------------------------------------------
"""

import torch.nn as nn

from models.baseline import Baseline
from models.losses import TripletLoss, CrossEntropyLabelSmooth

from models.DSA.DSA import DSA

MODEL = {
    'baseline': Baseline,
}


def make_model_dsa(cfg, num_classes):
    model = DSA(num_classes, dsag=True)

    return model


def make_loss_dsa(cfg, num_classes):
    if cfg.MODEL.LABEL_SMOOTH:
        xent_criterion = CrossEntropyLabelSmooth(num_classes=num_classes)
    else:
        xent_criterion = nn.CrossEntropyLoss()

    if cfg.SOLVER.LOSS == 'softmax_triplet':
        embedding_criterion = TripletLoss(margin=cfg.SOLVER.MARGIN)

        def criterion(softmax_y, triplet_y, labels):
            assert len(softmax_y) == 11 and len(triplet_y) == 2, "Error: Some loss items are missing."

            triplet_losses = [1.5 * embedding_criterion(output, labels)[0] for output in triplet_y]
            softmax_losses = []
            for idx, score in enumerate(softmax_y):
                if idx < 9:
                    softmax_losses.append(0.5 * xent_criterion(score, labels))
                else:
                    softmax_losses.append(xent_criterion(score, labels))
            sum_loss = triplet_losses + softmax_losses

            loss = sum(sum_loss)
            return loss

        return criterion
    else:
        raise KeyError("Unknown loss: ", cfg.SOLVER.LOSS)


def make_model(cfg, num_classes):
    if cfg.MODEL.NAME not in MODEL:
        raise KeyError("Unknown model: ", cfg.MODEL.NAME)
    else:
        model = MODEL[cfg.MODEL.NAME](num_classes, arch=cfg.MODEL.ARCH, stride=cfg.MODEL.STRIDE)

    return model


def make_loss(cfg, num_classes):
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
