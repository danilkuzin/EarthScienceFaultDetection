import pathlib

import torch
import torch.utils.data as data

import numpy as np

import h5py

from src.config import data_path

class h5_loader(data.Dataset):
    def __init__(self, filepaths, channels, transform=None):
        self.filepaths = filepaths
        self.channels = channels
        self.transform = transform

    def __parse_file__(self, f):
        with h5py.File(f, 'r') as hf:
            img = hf['img'][:].astype(np.float32)[:, :, self.channels]
            lbl = np.array(np.argmax(hf['lbl'][:])).astype(np.float32)
            coord = hf['coord'][:].astype(np.int32)
            return img, lbl

    def __getitem__(self, index):
        image, label = self.__parse_file__(self.filepaths[index])
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.filepaths)


class h5_loader_segmentation(data.Dataset):
    def __init__(self, filepaths, channels, transform=None):
        self.filepaths = filepaths
        self.channels = channels
        self.transform = transform

    def __parse_file__(self, f):
        with h5py.File(f, 'r') as hf:
            img = hf['img'][:].astype(np.float32)[:, :, self.channels]
            lbl = hf['lbl'][:].astype(np.int)
            coord = hf['coord'][:].astype(np.int32)
            return img, lbl

    def __getitem__(self, index):
        image, label = self.__parse_file__(self.filepaths[index])
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.filepaths)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}
