"""
-------------------------------------------------
   File Name:    finetune_cuhk01.py
   Author:       Zhonghao Huang
   Date:         2019/10/11
   Description:
-------------------------------------------------
"""

import os
import argparse

import torch
import torch.nn as nn
from torch.backends import cudnn

from trainers.trainer import Trainer
from data import make_loader, make_loader_cuhk01
from logger import make_logger
from models import make_model, make_loss
from optimizer import make_optimizer
from scheduler import make_scheduler


def train(cfg):
    # output
    output_dir = cfg.OUTPUT_DIR
    if os.path.exists(output_dir):
        raise KeyError("Existing path: ", output_dir)
    else:
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
    train_loader, query_loader, gallery_loader, num_classes = make_loader_cuhk01(cfg)

    # model
    model = make_model(cfg, num_classes=767)
    model.load_state_dict(
        torch.load(os.path.join('./checkpoint/CUHK03L/Exp-0922-1', model.__class__.__name__ + '_best.pth')))
    model.classifier = nn.Linear(1024, num_classes)

    if num_gpus > 1:
        model = nn.DataParallel(model)

    # solver
    criterion = make_loss(cfg, num_classes)
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
                out_dir=output_dir)

    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Person Re-identification Project.")
    parser.add_argument('--config', default='./configs/sample_CUHK01.yaml')
    args = parser.parse_args()

    from config import cfg as opt

    opt.merge_from_file(args.config)
    opt.freeze()

    train(opt)
