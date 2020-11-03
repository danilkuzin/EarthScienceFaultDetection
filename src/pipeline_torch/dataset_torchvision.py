import h5py
import numpy
import torch

from src.pipeline_torch.transforms_torchvision import ToTensor


class h5_loader_segmentation(torch.data.Dataset):
    def __init__(self, filepaths, channels, device, transform=None):
        self.filepaths = filepaths
        self.channels = channels
        self.device = device
        self.transform = transform

    def __parse_file__(self, f):
        with h5py.File(f, 'r') as hf:
            img = hf['img'][:].astype(numpy.float32)[:, :, self.channels]
            lbl = hf['lbl'][:].astype(numpy.float32)
            return img, lbl

    def __getitem__(self, index):
        image, label = self.__parse_file__(self.filepaths[index])
        sample = {'image': image, 'label': label}
        sample = ToTensor()(sample).to(self.device)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.filepaths)
