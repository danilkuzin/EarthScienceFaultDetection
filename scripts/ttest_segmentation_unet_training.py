import sys
import numpy as np
import torch

import torch.nn as nn

import torch.optim as optim
from torch.optim import lr_scheduler

import tensorflow as tf

import os

sys.path.extend(['../../EarthScienceFaultDetection'])
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

from src.pipeline_torch.dataset import ToTensor, RandomRotation, ToTFImageInput, \
    RandomHorizontalFlip, RandomVerticalFlip, RandomBrightness, RandomContrast, \
    FromTFImageOutput

from src.LearningTorch.net_architecture import UNet, LossBinary, FocalLoss, \
    LossCrossDice
from src.pipeline_torch.training import datasets_on_single_files_torch, \
    train_on_preloaded_single_files_torch_unet, \
    datasets_on_single_files_torch_segmentation

from src.pipeline.training import train_on_preloaded, load_data, datasets_on_single_files, \
    train_on_preloaded_single_files
from src.LearningKeras.net_architecture import cnn_150x150x5
from src.config import data_path

import torchvision

tf.enable_eager_execution()

np.random.seed(1000)

cnn_model = UNet(n_input_channels=5, n_classes=3)
criterion = nn.CrossEntropyLoss() # LossCrossDice(jaccard_weight=5) # FocalLoss(reduction='mean') # FocalLoss(gamma=0.5, alpha=0.5) # LossBinary(jaccard_weight=5) # nn.BCEWithLogitsLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=1e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10,
                                       gamma=0.1)

# im = np.random.randint(255, size=(1, 5, 156, 156)).astype(np.float32)
# im_tensor = torch.tensor(im)
#
# output = cnn_model(im_tensor)
# print(output.shape)


batch_size = 32
num_workers = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = torchvision.transforms.Compose([
    ToTFImageInput(),
    RandomRotation(),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomBrightness(),
    RandomContrast(),
    FromTFImageOutput(),
    ToTensor(device),
    ]
)

train_dataset, train_dataset_size, valid_dataset, valid_dataset_size = \
    datasets_on_single_files_torch_segmentation(
        device=device,
        regions=[6], path_prefix=f'{data_path}/train_data',
        channels=[0, 1, 2, 3, 4],
        train_ratio=0.80, batch_size=batch_size,
        num_workers=num_workers,
        transform=transform
)

train_on_preloaded_single_files_torch_unet(
    cnn_model, train_dataset, train_dataset_size, valid_dataset, valid_dataset_size,
    folder=f"{data_path}/results/test",
    epochs=10,
    batch_size=batch_size,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=exp_lr_scheduler)

