"""
-------------------------------------------------
   File Name:    train.py
   Author:       Zhonghao Huang
   Date:         2019/9/1
   Description:
-------------------------------------------------
"""

import os
import argparse

import torch
import torch.nn as nn
from torch.backends import cudnn

from trainers.trainer import Trainer
from data import make_loader
from logger import make_logger
from models import make_model, make_loss
from optimizer import make_optimizer
from scheduler import make_scheduler


def train(cfg):
    # output
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f_out:
        print(cfg, file=f_out)

    # logger
    logger = make_logger("project", output_dir, 'log')

    # device
    num_gpus = 0
    if cfg.DEVICE == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.DEVICE_ID
        num_gpus = len(cfg.DEVICE_ID.split(','))
        logger.info("Using {} GPUs.\n".format(num_gpus))
    cudnn.benchmark = True
    device = torch.device(cfg.DEVICE)

    # data
    train_loader, query_loader, gallery_loader, num_classes = make_loader(cfg)

    # model
    model = make_model(cfg, num_classes=num_classes)
    if num_gpus > 1:
        model = nn.DataParallel(model)

    # solver
    criterion = make_loss(cfg)
    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg, optimizer)

    # do_train
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      criterion=criterion,
                      logger=logger,
                      scheduler=scheduler,
                      device=device)

    trainer.run(start_epoch=0,
                total_epoch=cfg.SOLVER.MAX_EPOCHS,
                train_loader=train_loader,
                query_loader=query_loader,
                gallery_loader=gallery_loader,
                print_freq=cfg.SOLVER.PRINT_FREQ,
                eval_period=cfg.SOLVER.EVAL_PERIOD,
                checkpoint_period=cfg.SOLVER.CHECK_PERIOD)

    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Person Re-identification Project.")
    parser.add_argument('--config', default='./configs/sample.yaml')
    args = parser.parse_args()

    from config import cfg as opt

    opt.merge_from_file(args.config)
    opt.freeze()

    train(opt)
