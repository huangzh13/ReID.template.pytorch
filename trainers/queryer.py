"""
-------------------------------------------------
   File Name:    queryer.py
   Author:       Zhonghao Huang
   Date:         2019/10/6
   Description:
-------------------------------------------------
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch

from data import TestTransform


class Queryer:
    def __init__(self, model, query_loader, gallery_loader, device):
        self.model = model.to(device)
        self.query_loader = query_loader
        self.gallery_loader = gallery_loader
        self.transform = TestTransform()
        self.device = device

    def cal_dismat(self, idx):
        self.model.eval()
        query_sample = self.query_loader.dataset.dataset[idx]
        img, pid, camid = query_sample
        self.pid = pid
        self.cid = camid
        img = Image.open(img).convert('RGB')
        img = self.transform(img)
        img = torch.unsqueeze(img, dim=0)
        img = img.to(self.device)
        with torch.no_grad():
            qf = [self.model(img)]
        qf = torch.cat(qf, 0)

        gf, g_pids, g_cids = [], [], []
        for inputs in self.gallery_loader:
            imgs, pids, cids = inputs
            imgs = imgs.to(self.device)
            with torch.no_grad():
                feature = self.model(imgs)
            gf.append(feature)
            g_pids.extend(pids)
            g_cids.extend(cids)
        gf = torch.cat(gf, 0)
        print("Extracted features for gallery set: {} x {}".format(gf.size(0), gf.size(1)))

        print("Computing distance matrix")
        m, n = qf.size(0), gf.size(0)
        q_g_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                   torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        q_g_dist.addmm_(1, -2, qf, gf.t())
        self.distmat = q_g_dist.cpu().numpy()
        self.g_pids = g_pids
        self.g_cids = g_cids

    def query(self, idx, save_path='query_case'):
        self.cal_dismat(idx)

        indices = np.argsort(self.distmat, axis=1)

        fig, axes = plt.subplots(1, 11, figsize=(12, 8))
        img = self.query_loader.dataset.dataset[idx][0]
        pid = self.query_loader.dataset.dataset[idx][1]
        # camid = self.query_loader.dataset.dataset[i][2]
        img = Image.open(img).convert('RGB')
        axes[0].set_title(pid)
        axes[0].imshow(img)
        axes[0].set_axis_off()

        j = 0
        count = 0
        while j < 10:
            gallery_index = indices[0][count]

            count += 1
            if self.g_cids[gallery_index] == self.cid and self.g_pids[gallery_index] == self.pid:
                continue

            img = self.gallery_loader.dataset.dataset[gallery_index][0]
            pid = self.gallery_loader.dataset.dataset[gallery_index][1]
            img = Image.open(img).convert('RGB')
            axes[j + 1].set_title(pid)
            axes[j + 1].set_axis_off()
            axes[j + 1].imshow(img)
            j += 1

        fig.savefig(save_path + '_' + str(idx) + '.png')
        plt.close(fig)
