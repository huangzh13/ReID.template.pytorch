"""
-------------------------------------------------
   File Name:    trainer.py
   Author:       Zhonghao Huang
   Date:         2019/9/10
   Description:
-------------------------------------------------
"""

import time
import numpy as np

import torch

from utils import evaluation
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
            print_freq, eval_period, checkpoint_period):

        self.logger.info('Start at Epoch[{}]'.format(start_epoch))

        losses = AverageValueMeter()

        for epoch in range(start_epoch, total_epoch):
            if self.scheduler is not None:
                self.scheduler.step(epoch)

            self.model.train()

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

            # ===== Epoch done =====
            if (epoch + 1) % eval_period == 0:
                self.evaluator.evaluate(query_loader, gallery_loader)

            if (epoch + 1) % checkpoint_period == 0:
                print('Saved.\n')
