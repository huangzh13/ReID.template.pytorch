"""
-------------------------------------------------
   File Name:    test_dsa.py
   Author:       Zhonghao Huang
   Date:         2019/9/17
   Description:
-------------------------------------------------
"""

import os
import argparse

import torch
import torch.nn as nn
from torch.backends import cudnn

from data import make_loader_dsap, make_loader_flip_dsap
from models import make_model_dsa
from trainers.evaluator_dsa import EvaluatorDSA


def test(cfg, flip):
    # device
    num_gpus = 0
    if cfg.DEVICE == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.DEVICE_ID
        num_gpus = len(cfg.DEVICE_ID.split(','))
        print("Using {} GPUs.\n".format(num_gpus))
    cudnn.benchmark = True
    device = torch.device(cfg.DEVICE)

    # data
    train_loader, query_loader, gallery_loader, num_classes = make_loader_dsap(cfg)

    # model
    model = make_model_dsa(cfg, num_classes=num_classes)
    model.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, model.__class__.__name__ + '_best.pth')))
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    evaluator = EvaluatorDSA(model)

    # Results
    cmc, mAP = evaluator.evaluate(query_loader, gallery_loader)

    ranks = [1, 5, 10]
    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------\n")

    # Results with flip
    if flip:
        print("Results with flip --------------")
        query_flip_loader, gallery_flip_loader = make_loader_flip_dsap(cfg)
        cmc, mAP = evaluator.evaluate_flip(query_loader, gallery_loader, query_flip_loader, gallery_flip_loader)

        print("Results ----------")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("------------------\n")

    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Person Re-identification Project (DSA ReID).")
    parser.add_argument('--config', default='./configs/DSA.yaml')
    parser.add_argument('--flip', default=True)
    args = parser.parse_args()

    from config import cfg as opt

    opt.merge_from_file(args.config)
    opt.freeze()

    test(opt, args.flip)
