"""
-------------------------------------------------
   File Name:    DSAGStream.py
   Author:       Zhonghao Huang
   Date:         2019/7/8
   Description:
-------------------------------------------------
"""

import torch
import torch.nn as nn

from models.DSA.BaseModule import BasicBlock, BaseModule

__all__ = ['DSAGStream']


class FirstLayer(BaseModule):
    def __init__(self):
        super(FirstLayer, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.inplanes = 64
        self.layer_conv3 = self._make_layer(BasicBlock, 64, 2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer_conv3(x)

        return x


class SecondLayer(BaseModule):
    def __init__(self):
        super(SecondLayer, self).__init__()

        self.inplanes = 64
        self.layer_conv4 = self._make_layer(BasicBlock, 128, 2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer_conv4(x)

        return x


class MultiBranch(nn.Module):
    def __init__(self):
        super(MultiBranch, self).__init__()
        self.num_branch_1 = 24
        self.num_branch_2 = 13
        self.parts = 8

        for i in range(self.num_branch_1):
            name = 'layer_1_' + str(i)
            setattr(self, name, FirstLayer())
        for i in range(self.num_branch_2):
            name = 'layer_2_' + str(i)
            setattr(self, name, SecondLayer())

    @staticmethod
    def first_merging(feats):
        feats_merged = [
            feats[0],
            feats[1],
            feats[2] + feats[3],
            feats[4] + feats[5],
            feats[6] + feats[7],
            feats[8] + feats[9],
            feats[10] + feats[11],
            feats[12] + feats[13],
            feats[14] + feats[15],
            feats[16] + feats[17],
            feats[18] + feats[19],
            feats[20] + feats[21],
            feats[22] + feats[23]
        ]

        return feats_merged

    @staticmethod
    def second_merging(feats):
        feats_merged = [
            feats[0] + feats[1],
            feats[2],
            feats[3],
            feats[4] + feats[5],
            feats[6] + feats[7],
            feats[8] + feats[9],
            feats[10] + feats[11],
            feats[12]
        ]

        return feats_merged

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)

        feats_1 = []
        for i in range(self.num_branch_1):
            name = 'layer_1_' + str(i)
            sub_network = getattr(self, name)
            feats_1.append(sub_network(x[i]))

        feats_1_merged = self.first_merging(feats_1)

        feats_2 = []
        for i in range(self.num_branch_2):
            name = 'layer_2_' + str(i)
            sub_network = getattr(self, name)
            feats_2.append(sub_network(feats_1_merged[i]))

        feats_2_merged = self.second_merging(feats_2)
        # feats_2_merged = torch.cat(feats_2_merged, 1)

        return feats_2_merged


class HeadDSAG(BaseModule):
    def __init__(self):
        super(HeadDSAG, self).__init__()
        self.parts = 8

        self.inplanes = 1024
        self.layer_conv5_g = self._make_layer(BasicBlock, 2048, 2)

        for i in range(self.parts):
            name = 'layer_conv5_l_' + str(i)
            self.inplanes = 128
            setattr(self, name, self._make_layer(BasicBlock, 256, 2))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # global branch
        global_x = torch.cat(x, 1)
        global_feat = self.layer_conv5_g(global_x)
        global_feat = self.avg_pool(global_feat).squeeze()

        # local branch
        local_feats = []
        for i in range(self.parts):
            name = 'layer_conv5_l_' + str(i)
            conv5_l = getattr(self, name)
            feat = conv5_l(x[i])
            local_feats.append(self.avg_pool(feat).squeeze())

        return global_feat, local_feats


class DSAGStream(nn.Module):
    def __init__(self):
        super(DSAGStream, self).__init__()

        self.MB_Ns = MultiBranch()
        self.head = HeadDSAG()

    def forward(self, x):
        x = self.MB_Ns(x)
        global_feat, local_feats = self.head(x)

        return global_feat, local_feats


if __name__ == '__main__':
    # MB-Ns First layer
    dsap_image = torch.randn(32, 3, 32, 32)
    first = FirstLayer()
    out_1 = first(dsap_image)
    print(out_1.shape)

    # MB-Ns Second layer
    second = SecondLayer()
    out_2 = second(out_1)
    print(out_2.shape)

    # MultiBranch
    images = torch.randn(32, 24, 3, 32, 32)
    mb = MultiBranch()
    out_3 = mb(images)
    print(len(out_3))
    print(out_3[0].shape)

    # Head
    head = HeadDSAG()
    out_4 = head(out_3)
    print(out_4[0].shape)
    print(out_4[1][0].shape)

    # DSAG
    model = DSAGStream()
    out = model(images)
    print(out[0].shape)
    print(out[1][0].shape)
