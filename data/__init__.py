"""
-------------------------------------------------
   File Name:    __init__.py.py
   Author:       Zhonghao Huang
   Date:         2019/9/2
   Description:
-------------------------------------------------
"""

from torch.utils.data import DataLoader

from data.dataset import ReIDDataset, ImageData
from data.transforms import TrainTransform, TestTransform
from data.samplers import RandomIdentitySampler


def make_loader_dsap():
    pass


def make_loader(cfg):
    _data = ReIDDataset(dataset_dir=cfg.DATASETS.NAME, root=cfg.DATASETS.ROOT)

    train_loader = DataLoader(ImageData(_data.train, TrainTransform(p=0.5)),
                              sampler=RandomIdentitySampler(_data.train, cfg.DATALOADER.NUM_INSTANCES),
                              batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS,
                              pin_memory=True, drop_last=True)

    query_loader = DataLoader(ImageData(_data.query, TestTransform(flip=False)),
                              batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS,
                              pin_memory=True)

    gallery_loader = DataLoader(ImageData(_data.gallery, TestTransform(flip=False)),
                                batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS,
                                pin_memory=True)

    return train_loader, query_loader, gallery_loader
