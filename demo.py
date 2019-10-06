import argparse
import numpy as np
import os
import scipy.io
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import datasets

from data import make_loader
from models import make_model
from trainers.queryer import Queryer

matplotlib.use('agg')


#####################################################################
# Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


#######################################################################
# sort the images
def sort_img(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    # same camera
    camera_index = np.argwhere(gc == qc)

    # good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index


def query(cfg, query_index):
    # device
    num_gpus = 0
    if cfg.DEVICE == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.DEVICE_ID
        num_gpus = len(cfg.DEVICE_ID.split(','))
        print("Using {} GPUs.\n".format(num_gpus))
    device = torch.device(cfg.DEVICE)

    # data
    train_loader, query_loader, gallery_loader, num_classes = make_loader(cfg)

    # model
    model = make_model(cfg, num_classes=num_classes)
    model.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, model.__class__.__name__ + '_best.pth')))
    if num_gpus > 1:
        model = nn.DataParallel(model)

    queryer = Queryer(model=model, query_loader=query_loader, gallery_loader=gallery_loader, device=device)
    queryer.query(idx=query_index)

    print('Done.')


if __name__ == '__main__':
    ######################################################################
    # Evaluate
    parser = argparse.ArgumentParser(description='Demo for visualization.')
    parser.add_argument('--query_index', default=177, type=int, help='test_image_index')
    parser.add_argument('--config', default='./configs/sample.yaml')
    args = parser.parse_args()

    from config import cfg as opt

    opt.merge_from_file(args.config)
    opt.freeze()

    query(opt, args.query_index)

    print('Done.')
