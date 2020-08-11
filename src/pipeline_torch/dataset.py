import pathlib

import torch
import torch.utils.data as data

import tensorflow as tf

from torchvision.transforms import functional
import random

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
            lbl = hf['lbl'][:].astype(np.float32)
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


class ToTFImageInput(object):
    """Convert input into compatible with tf.image"""

    def __init__(self):
        pass

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        label = np.expand_dims(label, axis=2)
        return {'image': image,
                'label': label}


class FromTFImageOutput(object):
    """Convert output of tf.image back to numpy arrays"""

    def __init__(self):
        pass

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = image.numpy()
        label = label.numpy()
        label = np.squeeze(label)
        return {'image': image,
                'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, device):
        self.device = device

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).to(self.device),
                'label': torch.from_numpy(label).to(self.device)}


class RandomRotation(object):
    """Rotate on random number of 90 degree rotation"""

    def __init__(self):
        pass

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k)
        label = tf.image.rot90(label, k)

        return {'image': image,
                'label': label}


class RandomHorizontalFlip(object):
    """Randomly (with 50% chance) flip input horizontally"""

    def __init__(self):
        pass

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random > 0.5:
            image = tf.image.flip_left_right(image)
            label = tf.image.flip_left_right(label)

        return {'image': image,
                'label': label}


class RandomVerticalFlip(object):
    """Randomly (with 50% chance) flip input vertically"""

    def __init__(self):
        pass

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random > 0.5:
            image = tf.image.flip_up_down(image)
            label = tf.image.flip_up_down(label)

        return {'image': image,
                'label': label}


class RandomBrightness(object):
    """Adjust the brightness of images by a random factor"""

    def __init__(self):
        pass

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = tf.image.random_brightness(image, 0.05)

        return {'image': image,
                'label': label}


class RandomContrast(object):
    """Adjust the contrast of an image or images by a random factor"""

    def __init__(self):
        pass

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = tf.image.random_contrast(image, 0.7, 1.3)

        return {'image': image,
                'label': label}


