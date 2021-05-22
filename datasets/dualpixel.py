from typing import Tuple
import os
import glob
import torch
import numpy as np
import skimage.io
from torch.utils.data import Dataset
from datasets.augmentation import RandomTransform
from kornia.augmentation import CenterCrop
import cv2
import skimage.transform
from util.helper import ips_to_metric, metric_to_ips

CROP_WIDTH = 20
TRAIN_BASE_DIR = os.path.join('data', 'training_data', 'dualpixel', 'train')
TEST_BASE_DIR = os.path.join('data', 'training_data', 'dualpixel', 'test')


def get_captures(base_dir):
    """Gets a list of captures."""
    depth_dir = os.path.join(base_dir, 'inpainted_depth')
    return [
        name for name in os.listdir(depth_dir)
        if os.path.isdir(os.path.join(depth_dir, name))
    ]


class DualPixel(Dataset):

    def __init__(self, dataset: str, image_size: Tuple[int, int], is_training: bool = True, randcrop: bool = False,
                 augment: bool = False, padding: int = 0, upsample_factor: int = 2):
        super().__init__()
        if dataset == 'train':
            base_dir = TRAIN_BASE_DIR
        elif dataset == 'val':
            base_dir = TEST_BASE_DIR
        else:
            raise ValueError(f'dataset ({dataset}) has to be "train," "val," or "example."')

        self.transform = RandomTransform(image_size, randcrop, augment)
        self.centercrop = CenterCrop(image_size)

        captures = get_captures(base_dir)
        self.sample_ids = []
        for id in captures:
            image_path = glob.glob(os.path.join(base_dir, 'scaled_images', id, '*_center.jpg'))[0]
            depth_path = glob.glob(os.path.join(base_dir, 'inpainted_depth', id, '*_center.png'))[0]
            conf_path = glob.glob(os.path.join(base_dir, 'merged_conf', id, '*_center.exr'))[0]
            sample_id = {
                'image_path': image_path,
                'depth_path': depth_path,
                'conf_path': conf_path,
                'id': id,
            }
            self.sample_ids.append(sample_id)
        self.min_depth = 0.2
        self.max_depth = 100.
        self.is_training = is_training
        self.padding = padding
        self.upsample_factor = upsample_factor

    def stretch_depth(self, depth, depth_range, min_depth):
        return depth_range * depth + min_depth

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        image_path = sample_id['image_path']
        depth_path = sample_id['depth_path']
        conf_path = sample_id['conf_path']
        id = sample_id['id']

        depth = skimage.io.imread(depth_path).astype(np.float32)[..., None] / 255
        img = skimage.io.imread(image_path).astype(np.float32) / 255
        conf = cv2.imread(filename=conf_path, flags=-1)[..., [2]]

        depth = depth[CROP_WIDTH:-CROP_WIDTH, CROP_WIDTH:-CROP_WIDTH, :]
        img = img[CROP_WIDTH:-CROP_WIDTH, CROP_WIDTH:-CROP_WIDTH, :]
        conf = conf[CROP_WIDTH:-CROP_WIDTH, CROP_WIDTH:-CROP_WIDTH, :]

        img = np.pad(img,
                     ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='reflect')
        depth = np.pad(depth,
                       ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='reflect')
        conf = np.pad(conf,
                      ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='reflect')

        if self.upsample_factor != 1:
            img = skimage.transform.rescale(img, self.upsample_factor, multichannel=True, order=3)
            depth = skimage.transform.rescale(depth, self.upsample_factor, multichannel=True, order=3)
            conf = skimage.transform.rescale(conf, self.upsample_factor, multichannel=True, order=3)

        img = torch.from_numpy(img).permute(2, 0, 1)
        depthmap = torch.from_numpy(depth).permute(2, 0, 1)
        conf = torch.from_numpy(conf).permute(2, 0, 1)

        depthmap_metric = ips_to_metric(depthmap, self.min_depth, self.max_depth)
        if depthmap_metric.min() < 1.0:
            depthmap_metric += (1. - depthmap_metric.min())
        depthmap = metric_to_ips(depthmap_metric.clamp(1.0, 5.0), 1.0, 5.0)


        if self.is_training:
            img, depthmap, conf = self.transform(img, depthmap, conf)
        else:
            img = self.centercrop(img)
            depthmap = self.centercrop(depthmap)
            conf = self.centercrop(conf)

        # Remove batch dim (Kornia adds batch dimension automatically.)
        img = img.squeeze(0)
        depthmap = depthmap.squeeze(0)
        depth_conf = torch.where(conf.squeeze(0) > 0.99, 1., 0.)

        sample = {'id': id, 'image': img, 'depthmap': depthmap, 'depth_conf': depth_conf}

        return sample
