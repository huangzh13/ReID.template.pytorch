"""
-------------------------------------------------
   File Name:    baseline.py
   Author:       Zhonghao Huang
   Date:         2019/9/9
   Description:
-------------------------------------------------
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet101

from models.blocks import weights_init_kaiming, weights_init_classifier

# arch type
FACTORY = {
    'resnet50': resnet50,
    'resnet101': resnet101
}


class Baseline(nn.Module):
    def __init__(self, num_classes, arch='resnet50', stride=1):
        super(Baseline, self).__init__()
        self.num_classes = num_classes
        self.arch = arch
        self.stride = stride

        # backbone
        if arch not in FACTORY:
            raise KeyError("Unknown arch: ", arch)
        else:
            resnet = FACTORY[arch](pretrained=True)
            if stride == 1:
                resnet.layer4[0].downsample[0].stride = (1, 1)
                resnet.layer4[0].conv2.stride = (1, 1)

            self.backbone = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,  # res_conv2
                resnet.layer2,  # res_conv3
                resnet.layer3,  # res_conv4
                resnet.layer4
            )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.bottleneck = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.5)
        )
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(1024, self.num_classes)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        global_feat = self.backbone(x)
        global_feat = self.gap(global_feat)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)

        if self.training:
            feat = self.bottleneck(global_feat)
            cls_score = self.classifier(feat)
            return [cls_score], [global_feat]
        else:
            return global_feat


if __name__ == '__main__':
    data = torch.rand(32, 3, 384, 128)

    model = Baseline(num_classes=751, arch='resnet101', stride=1)
    out = model(data)
    print(out[0][0].shape)
    print(out[1][0].shape)

    model = Baseline(num_classes=751, arch='resnet101', stride=2)
    out = model(data)
    print(out[0][0].shape)
    print(out[1][0].shape)

    print('Done.')
