"""
-------------------------------------------------
   File Name:    DSA.py
   Author:       Zhonghao Huang
   Date:         2019/7/8
   Description:
-------------------------------------------------
"""

import math
import torch
import torch.nn as nn

from models.DSA.DSAGStream import DSAGStream
from models.DSA.MFStream import MFStream
from models.blocks import BNNeck

__all__ = ['DSA']


class DSA(nn.Module):
    def __init__(self, n_class, dsag=True):
        super(DSA, self).__init__()

        self.n_class = n_class
        self.MFStream = MFStream()
        self.dsag = dsag
        if dsag:
            self.DSAGStream = DSAGStream()
        self.parts = 8

        self.classifier_global_mf = BNNeck(in_planes=2048, num_classes=self.n_class)

        self.classifier_global_fused = BNNeck(in_planes=2048, num_classes=self.n_class)
        self.classifier_local_fused = BNNeck(in_planes=2048, num_classes=self.n_class)

        for i in range(self.parts):
            classifier_name = 'classifier_local_mf_' + str(i)
            setattr(self, classifier_name, BNNeck(in_planes=256, num_classes=self.n_class))

    def forward(self, x, bnneck=True, infer_dsap=True):
        img_ori, img_dsap = x

        global_feat_mf, local_feats_mf = self.MFStream(img_ori)

        if self.dsag and infer_dsap:
            global_feat_dsag, local_feats_dsag = self.DSAGStream(img_dsap)

            global_feat_fused = global_feat_mf + global_feat_dsag
            local_feats_fused = [local_feats_mf[i] + local_feats_dsag[i] for i in range(self.parts)]
            local_feats_fused_cat = torch.cat(local_feats_fused, 1)
        else:
            global_feat_fused = global_feat_mf
            local_feats_fused_cat = torch.cat(local_feats_mf, 1)

        if self.training:
            # ID loss for feats from MFStream
            cls_mf = [self.classifier_global_mf(global_feat_mf)]
            for i in range(self.parts):
                classifier_name = 'classifier_local_mf_' + str(i)
                classifier = getattr(self, classifier_name)
                cls_mf.append(classifier(local_feats_mf[i]))

            # ID loss and Triplet loss for fused feats
            cls_fused = [self.classifier_global_fused(global_feat_fused),
                         self.classifier_local_fused(local_feats_fused_cat)]

            return cls_mf + cls_fused, [global_feat_fused, local_feats_fused_cat]
        else:
            if bnneck:
                local_feat = self.classifier_local_fused.bn(local_feats_fused_cat)
                global_feat = self.classifier_global_fused.bn(global_feat_fused)
            else:
                local_feat = local_feats_fused_cat
                global_feat = global_feat_fused

            local_feat_norm = nn.functional.normalize(local_feat) * (1 / math.sqrt(2))
            global_feat_norm = nn.functional.normalize(global_feat) * (1 / math.sqrt(2))
            feat = torch.cat([global_feat_norm, local_feat_norm], 1)
            return feat


if __name__ == '__main__':
    images_ori = torch.randn(32, 3, 384, 128)
    images_dsap = torch.randn(32, 24, 3, 32, 32)

    model_1 = DSA(751, True)
    out = model_1((images_ori, images_dsap))
    print(out[0][0].shape)
    print(len(out[0]))
    print(out[1][0].shape)
    print(len(out[1]))
    print(out[2][0].shape)
    print(len(out[2]))

    print('Eval...')
    model_1.eval()
    out = model_1((images_ori, images_dsap))
    print(len(out))
    out = model_1((images_ori, images_dsap), bnneck=False)
    print(len(out))
    out = model_1((images_ori, images_dsap), bnneck=False, infer_dsap=False)
    print(len(out))
    print('Done.')

    model_2 = DSA(751, False)
    out = model_2((images_ori, images_dsap))
    print(out[0][0].shape)
    print(out[1][0].shape)
    print(out[2][0].shape)
