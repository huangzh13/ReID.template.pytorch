"""
-------------------------------------------------
   File Name:    __init__.py.py
   Author:       Zhonghao Huang
   Date:         2019/9/2
   Description:
-------------------------------------------------
"""

from torch.utils.data import DataLoader

from data.cuhk01 import CUHK01
from data.dataset import ReIDDataset, ImageData
from data.transforms import TrainTransform, TestTransform
from data.samplers import RandomIdentitySampler


def make_loader_flip(cfg):
    _data = ReIDDataset(dataset_dir=cfg.DATASETS.NAME, root=cfg.DATASETS.ROOT)

    query_flip_loader = DataLoader(ImageData(_data.query, TestTransform(flip=True)),
                                   batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS,
                                   pin_memory=True)

    gallery_flip_loader = DataLoader(ImageData(_data.gallery, TestTransform(flip=True)),
                                     batch_size=cfg.DATALOADER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS,
                                     pin_memory=True)

    return query_flip_loader, gallery_flip_loader


def make_loader(cfg):
    _data = ReIDDataset(dataset_dir=cfg.DATASETS.NAME, root=cfg.DATASETS.ROOT)
    num_train_pids = _data.num_train_pids

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

    return train_loader, query_loader, gallery_loader, num_train_pids


def make_loader_cuhk01(cfg):
    _data = CUHK01(root=cfg.DATASETS.ROOT)
    num_train_pids = _data.num_train_pids

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

    return train_loader, query_loader, gallery_loader, num_train_pids
