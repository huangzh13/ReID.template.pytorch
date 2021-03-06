"""
-------------------------------------------------
   File Name:    trainer.py
   Author:       Zhonghao Huang
   Date:         2019/9/10
   Description:
-------------------------------------------------
"""

import os
import time

import torch

from utils.meters import AverageValueMeter
from trainers.evaluator import Evaluator


class Trainer:
    def __init__(self, model, optimizer, criterion, logger, device, scheduler=None):
        self.model = model.to(device)
        self.evaluator = Evaluator(self.model)
        self.optimizer = optimizer
        self.criterion = criterion
        self.logger = logger
        self.device = device
        self.scheduler = scheduler

        self.step = 0

    def _parse_data(self, inputs):
        imgs, pids, _ = inputs

        self.data = imgs.to(self.device)
        self.target = pids.to(self.device)

    def run(self, start_epoch, total_epoch, train_loader, query_loader, gallery_loader,
            print_freq, eval_period, out_dir):

        self.logger.info('Start at Epoch[{}]\n'.format(start_epoch))

        losses = AverageValueMeter()

        best_epoch = 0
        best_mAP = 0.
        for epoch in range(start_epoch, total_epoch):
            self.model.train()
            losses.reset()

            if self.scheduler is not None:
                self.scheduler.step(epoch)
                self.logger.info('Epoch[{}] Lr:{:.2e}'.format(epoch, self.scheduler.get_lr()[0]))

            start = time.time()
            interval = len(train_loader) // print_freq
            for batch_index, inputs in enumerate(train_loader):
                # model optimizer
                self._parse_data(inputs)
                score, feat = self.model(self.data)
                self.loss = self.criterion(score, feat, self.target)
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                losses.add(self.loss.item())

                # logging
                if batch_index % interval == 0:
                    self.logger.info(
                        'Epoch[{}] Iteration[{}/{}] Loss: {:.4f}'.format(epoch, batch_index + 1,
                                                                         len(train_loader),
                                                                         losses.value()[0]))
            self.logger.info('Epoch[{}] Done.\n'.format(epoch))

            # ===== Epoch done =====
            if (epoch + 1) % eval_period == 0:
                ranks = [1, 5, 10]

                cmc, mAP = self.evaluator.evaluate(query_loader, gallery_loader)

                self.logger.info("Results ----------")
                self.logger.info("mAP: {:.1%}".format(mAP))
                self.logger.info("CMC curve")
                for r in ranks:
                    self.logger.info("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
                self.logger.info("------------------\n")

                if mAP > best_mAP:
                    best_mAP = mAP
                    best_epoch = epoch

                    save_filename = (self.model.__class__.__name__ + '_best.pth')
                    torch.save(self.model.state_dict(), os.path.join(out_dir, save_filename))
                    self.logger.info(save_filename + ' saved.\n')

        self.logger.info('Best mAP {:.1%}, achieved at Epoch [{}]'.format(best_mAP, best_epoch))
