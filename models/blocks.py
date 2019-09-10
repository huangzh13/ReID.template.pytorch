"""
-------------------------------------------------
   File Name:    blocks.py
   Author:       Zhonghao Huang
   Date:         2019/9/10
   Description:
-------------------------------------------------
"""

import torch.nn as nn


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, linear=True, num_bottleneck=512,
                 bn=True, relu=False, drop=0.5, return_feat=False):
        super(ClassBlock, self).__init__()
        self.return_feat = return_feat
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bn:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if drop > 0:
            add_block += [nn.Dropout(p=drop)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        fc = nn.Linear(num_bottleneck, class_num)
        fc.apply(weights_init_classifier)

        self.add_block = add_block
        self.fc = fc

    def forward(self, x):
        x = self.add_block(x)
        if self.return_feat:
            feat = x
            cls = self.fc(x)
            return cls, feat
        else:
            cls = self.fc(x)
            return cls


class BNNeck(nn.Module):
    def __init__(self, in_planes, num_classes):
        super(BNNeck, self).__init__()
        self.in_planes = in_planes
        self.num_classes = num_classes
        self.bn = nn.BatchNorm1d(self.in_planes)
        self.bn.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.in_planes, self.num_classes)

        self.bn.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        feat = self.bn(x)
        cls = self.classifier(feat)

        if self.training:
            return cls, x
        else:
            return cls, feat
