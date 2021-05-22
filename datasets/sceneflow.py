from typing import Tuple
import os
import torch
import numpy as np
import imageio
from torch.utils.data import Dataset
from datasets.augmentation import RandomTransform
from kornia.augmentation import CenterCrop
from kornia.filters import gaussian_blur2d

# 'left' image has negative disparity.
DATA_ROOT = os.path.join('data', 'training_data', 'SceneFlow')
TRAIN_IMAGE_PATH = [
    os.path.join(DATA_ROOT, 'FlyingThings3D_subset', 'train', 'image_clean', s) for s in ['right']
]
TRAIN_DISPARITY_PATH = [
    os.path.join(DATA_ROOT, 'FlyingThings3D_subset_disparity', 'train', 'disparity', s) for s in ['right']
]
VALIDATION_IMAGE_PATH = [
    os.path.join(DATA_ROOT, 'FlyingThings3D_subset', 'val', 'image_clean', s) for s in ['right']
]
VALIDATION_DISPARITY_PATH = [
    os.path.join(DATA_ROOT, 'FlyingThings3D_subset_disparity', 'val', 'disparity', s) for s in ['right']
]
EXAMPLE_IMAGE_PATH = [
    os.path.join(DATA_ROOT, 'example', 'FlyingThings3D', 'RGB_cleanpass', 'left')
]
EXAMPLE_DISPARITY_PATH = [
    os.path.join(DATA_ROOT, 'example', 'FlyingThings3D', 'disparity')
]


class SceneFlow(Dataset):

    def __init__(self, dataset: str, image_size: Tuple[int, int], is_training: bool = True, randcrop: bool = False,
                 augment: bool = False, padding: int = 0, singleplane: bool = False, n_depths: int = 16):
        """
        SceneFlow dataset is downloaded from
        https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
        Virtual image sensor size: 960 px x 540 px  or 32mm x 18mm
        Virtual focal length: 35mmx
        Baseline: 1 Blender unit
        """
        super().__init__()
        if dataset == 'train':
            image_dirs = TRAIN_IMAGE_PATH
            disparity_dirs = TRAIN_DISPARITY_PATH
        elif dataset == 'val':
            image_dirs = VALIDATION_IMAGE_PATH
            disparity_dirs = VALIDATION_DISPARITY_PATH
        elif dataset == 'example':
            image_dirs = EXAMPLE_IMAGE_PATH
            disparity_dirs = EXAMPLE_DISPARITY_PATH
        else:
            raise ValueError(f'dataset ({dataset}) has to be "train," "val," or "example."')

        self.transform = RandomTransform(image_size, randcrop, augment)
        self.centercrop = CenterCrop(image_size)

        self.sample_ids = []
        for image_dir, disparity_dir in zip(image_dirs, disparity_dirs):
            for filename in sorted(os.listdir(image_dir)):
                if '.png' in filename:
                    id = os.path.splitext(filename)[0]
                    disparity_path = os.path.join(disparity_dir, f'{id}.pfm')
                    if os.path.exists(disparity_path):
                        sample_id = {
                            'image_dir': image_dir,
                            'disparity_dir': disparity_dir,
                            'id': id,
                        }
                        self.sample_ids.append(sample_id)
                    else:
                        print(f'Disparity image does not exist!: {disparity_path}')
        self.is_training = torch.tensor(is_training)
        self.padding = padding
        self.singleplane = torch.tensor(singleplane)
        self.n_depths = n_depths

    def stretch_depth(self, depth, depth_range, min_depth):
        return depth_range * depth + min_depth

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        image_dir = sample_id['image_dir']
        disparity_dir = sample_id['disparity_dir']
        id = sample_id['id']

        disparity = np.flip(imageio.imread(os.path.join(disparity_dir, f'{id}.pfm')), axis=0).astype(np.float32)
        img = imageio.imread(os.path.join(image_dir, f'{id}.png')).astype(np.float32)
        img /= 255.  # Scale to [0, 1]

        img = np.pad(img,
                     ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='reflect')
        disparity = np.pad(disparity,
                           ((self.padding, self.padding), (self.padding, self.padding)), mode='reflect')

        img = torch.from_numpy(img).permute(2, 0, 1)
        disparity = torch.from_numpy(disparity)[None, ...]

        # A far object is 0.
        depthmap = disparity
        depthmap -= depthmap.min()
        depthmap /= depthmap.max()

        # Flip the value. A near object is 0.
        depthmap = 1. - depthmap

        if self.is_training:
            img, depthmap = self.transform(img, depthmap)
        else:
            img = self.centercrop(img)
            depthmap = self.centercrop(depthmap)

        # SceneFlow's depthmap has some aliasing artifact.
        depthmap = gaussian_blur2d(depthmap, sigma=(0.8, 0.8), kernel_size=(5, 5))

        # Remove batch dim (Kornia adds batch dimension automatically.)
        img = img.squeeze(0)
        depthmap = depthmap.squeeze(0)

        if self.singleplane:
            if self.is_training:
                depthmap = torch.rand((1,), device=depthmap.device) * torch.ones_like(depthmap)
            else:
                depthmap = torch.linspace(0., 1., steps=self.n_depths)[idx % self.n_depths] * torch.ones_like(depthmap)



        sample = {'id': id, 'image': img, 'depthmap': depthmap, 'depth_conf': torch.ones_like(depthmap)}

        return sample
