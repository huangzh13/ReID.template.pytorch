"""
-------------------------------------------------
   File Name:    baseline.py
   Author:       Zhonghao Huang
   Date:         2019/9/9
   Description:
-------------------------------------------------
"""

import torch.nn as nn


class Baseline(nn.Module):
    def __init__(self, num_classes, backbone, pretrain_choice):
        super(Baseline, self).__init__()
        self.pretrain_choice = pretrain_choice

    def forward(self, x):
        return x
