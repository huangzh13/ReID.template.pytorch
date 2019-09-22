"""
-------------------------------------------------
   File Name:    MFStream.py
   Author:       Zhonghao Huang
   Date:         2019/7/6
   Description:
-------------------------------------------------
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50

from models.DSA.BaseModule import Bottleneck, BaseModule

__all__ = ['MFStream']


class HeadMF(BaseModule):
    """
    The Head network of MF-Stream.
    """

    def __init__(self):
        super(HeadMF, self).__init__()
        self.parts = 8

        # local branch (8)
        for i in range(self.parts):
            name = 'layer_conv5_l_' + str(i)
            self.inplanes = 128
            setattr(self, name, self._make_layer(Bottleneck, 64, 3, stride=2))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # global branch
        self.inplanes = 1024
        resnet = resnet50(pretrained=True)
        # self.layer_conv5_g = self._make_layer(Bottleneck, 512, 3, stride=2)
        self.layer_conv5_g = resnet.layer4

    def forward(self, x):
        # global branch
        global_feat = self.layer_conv5_g(x)
        global_feat = self.avg_pool(global_feat).squeeze()

        # local branch
        local_feats = []
        for i in range(self.parts):
            name = 'layer_conv5_l_' + str(i)
            conv5_l = getattr(self, name)
            feat = conv5_l(x[:, i * 128:(i + 1) * 128, :, :])
            local_feats.append(self.avg_pool(feat).squeeze())

        return global_feat, local_feats


class MFStream(nn.Module):

    def __init__(self, pretrained=True):
        super(MFStream, self).__init__()

        resnet = resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # res_conv2
            resnet.layer2,  # res_conv3
            resnet.layer3,  # res_conv4
        )

        self.head = HeadMF()

    def forward(self, x):
        feat_map = self.backbone(x)

        global_feat, local_feats = self.head(feat_map)

        return global_feat, local_feats


if __name__ == '__main__':
    images_ori = torch.randn(32, 3, 256, 128)
    model = MFStream(True)

    out = model(images_ori)
    print(out[0].shape)
    print(out[1][0].shape)
