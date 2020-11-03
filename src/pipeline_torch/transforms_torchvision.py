import numpy as np
import torch
import torchvision

"""
This is a wrapper for torchvision.transforms for multi-target transformations. 
See also torchvision segmentation transforms as examples
"""

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {
            'image': torchvision.transforms.functional.to_tensor(image),
            'label': torch.as_tensor(np.array(label), dtype=torch.int32)
        }

# todo add resize, randomcrop

class RandomHorizontalFlip(torch.nn.Module):
    """Randomly flip input horizontally"""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, sample):
        image, label = sample['image'], sample['label']

        if torch.rand(1) < self.p:
            image = torchvision.transforms.functional.hflip(image)
            label = torchvision.transforms.functional.hflip(label)

        return {'image': image, 'label': label}


class RandomVerticalFlip(torch.nn.Module):
    """Randomly flip input vertically"""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, sample):
        image, label = sample['image'], sample['label']

        if torch.rand(1) < self.p:
            image = torchvision.transforms.functional.vflip(image)
            label = torchvision.transforms.functional.vflip(label)

        return {'image': image, 'label': label}




# todo RandomPerspective
# todo RandomResizedCrop
# todo RandomSizedCrop

class ColorJitter(torchvision.transforms.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def forward(self, sample):
        return {
            'image': super(ColorJitter, self).forward(sample['image']),
            'label': sample['label']
        }


class RandomRotation(torch.nn.Module):
    """Rotate on random rotation"""

    def __init__(self, degrees):
        super().__init__()
        self.degrees = degrees

    def forward(self, sample):
        image, label = sample['image'], sample['label']

        angle = torchvision.transforms.RandomRotation.get_params(degrees=self.degrees)
        image = torchvision.transforms.functional.rotate(img=image, angle=angle)
        label = torchvision.transforms.functional.rotate(img=label, angle=angle)

        return {'image': image, 'label': label}


# todo RandomAffine
