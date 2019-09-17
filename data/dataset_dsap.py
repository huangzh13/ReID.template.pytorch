"""
-------------------------------------------------
   File Name:    dataset_dsap.py
   Author:       Zhonghao Huang
   Date:         2019/9/17
   Description:
-------------------------------------------------
"""

import re
import os

from torch.utils.data import Dataset

from data.dataset import read_image


class ImageDataDSAP(Dataset):
    def __init__(self, dataset, transform, transform_dsap):
        self.dataset = dataset
        self.transform = transform
        self.transform_dsap = transform_dsap

    def __getitem__(self, item):
        img, img_dsap, pid, cam_id = self.dataset[item]
        img = read_image(img)
        img_dsap = read_image(img_dsap)
        if self.transform is not None:
            img = self.transform(img)
            img_dsap = self.transform_dsap(img_dsap)
        return img, img_dsap, pid, cam_id

    def __len__(self):
        return len(self.dataset)


class ReIDDatasetDSAP:
    """
    """

    def __init__(self, dataset_dir, root='/home/hzh/data', mode='retrieval'):
        self.dataset_dir = dataset_dir
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = os.path.join(self.dataset_dir, 'query')
        self.gallery_dir = os.path.join(self.dataset_dir, 'bounding_box_test')

        self.train_dir_dsap = os.path.join(self.dataset_dir, 'DSAP', 'bounding_box_train')
        self.query_dir_dsap = os.path.join(self.dataset_dir, 'DSAP', 'query')
        self.gallery_dir_dsap = os.path.join(self.dataset_dir, 'DSAP', 'bounding_box_test')

        self._check_before_run()
        train_relabel = (mode == 'retrieval')
        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, self.train_dir_dsap,
                                                                  relabel=train_relabel)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, self.query_dir_dsap, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, self.gallery_dir_dsap,
                                                                        relabel=False)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> {:s} loaded".format(self.dataset_dir))
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not os.path.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not os.path.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

        if not os.path.exists(self.train_dir_dsap):
            raise RuntimeError("'{}' is not available".format(self.train_dir_dsap))
        if not os.path.exists(self.query_dir_dsap):
            raise RuntimeError("'{}' is not available".format(self.query_dir_dsap))
        if not os.path.exists(self.gallery_dir_dsap):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir_dsap))

    @staticmethod
    def _process_dir(dir_path, dsap_dir_path, relabel=False):
        img_names = os.listdir(dir_path)
        img_paths = [os.path.join(dir_path, img_name) for img_name in img_names
                     if img_name.endswith('jpg') or img_name.endswith('png')]
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            # assert 0 <= pid <= 1501  # pid == 0 means background
            # assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            img_name = img_path.split('/')[-1]
            dsap_path = os.path.join(dsap_dir_path, img_name)
            if not os.path.exists(dsap_path):
                dsap_path = os.path.join(dsap_dir_path, 'dsap_lost.jpg')
            dataset.append((img_path, dsap_path, pid, camid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs


if __name__ == '__main__':
    market_dsap_data = ReIDDatasetDSAP(dataset_dir='Market1501')
    cuhk03l_dsap_data = ReIDDatasetDSAP(dataset_dir='CUHK03/labeled')

    print('Done.')
