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

from models.blocks import weights_init_kaiming, weights_init_classifier, ClassBlock

# arch type
FACTORY = {
    'resnet50': resnet50,
    'resnet101': resnet101
}


class PCB(nn.Module):
    def __init__(self, num_classes, arch='resnet50', stride=1):
        super(PCB, self).__init__()
        self.num_classes = num_classes
        self.part = 6

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

        self.pool_c = nn.Sequential(nn.AdaptiveAvgPool2d(self.part, 1),
                                    nn.Dropout(p=0.5))
        self.pool_e = nn.AdaptiveAvgPool2d((self.part, 1))

        # classifiers
        for i in range(self.part):
            name = 'classifier_' + str(i)
            setattr(self, name, ClassBlock(2048, self.num_classes, drop=0, num_bottleneck=256))

        # embedding
        for i in range(self.part):
            name = 'embed_' + str(i)
            setattr(self, name, nn.Linear(2048, 256))

    def forward(self, x):
        feats = self.backbone(x)
        feats_c = torch.squeeze(self.pool_c(feats))
        feats_e = torch.squeeze(self.pool_e(feats))

        logits_list = []
        for i in range(self.part_num):
            features_i = torch.squeeze(feats_c[:, :, i])
            classifier_i = getattr(self, 'classifier_' + str(i))
            logits_i = classifier_i(features_i)
            logits_list.append(logits_i)

        embeddings_list = []
        for i in range(self.part_num):
            feats_i = torch.squeeze(feats_e[:, :, i])
            embed_i = getattr(self, 'embed_' + str(i))
            embedding_i = embed_i(feats_i)
            embeddings_list.append(embedding_i)

        return logits_list, embeddings_list


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
    data = torch.rand(32, 3, 256, 128)

    model_pcb = PCB(num_classes=751)
    out = model_pcb(data)
    print(out[0][0].shape)
    print(out[1][0].shape)
    model_pcb.eval()
    out = model_pcb(data)
    print(out[0].shape)

    model = Baseline(num_classes=751, arch='resnet101', stride=1)
    out = model(data)
    print(out[0][0].shape)
    print(out[1][0].shape)

    model = Baseline(num_classes=751, arch='resnet101', stride=2)
    out = model(data)
    print(out[0][0].shape)
    print(out[1][0].shape)

    print('Done.')
