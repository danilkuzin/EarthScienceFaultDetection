import sys
import numpy as np
import torch

import torch.nn as nn

import torch.optim as optim
from torch.optim import lr_scheduler

sys.path.extend(['../../EarthScienceFaultDetection'])

from src.LearningTorch.net_architecture import FCNet
from src.pipeline_torch.training import datasets_on_single_files_torch, \
    train_on_preloaded_single_files_torch, \
    datasets_on_single_files_torch_segmentation

from src.pipeline.training import train_on_preloaded, load_data, datasets_on_single_files, \
    train_on_preloaded_single_files
from src.LearningKeras.net_architecture import cnn_150x150x5
import tensorflow as tf
from src.config import data_path

tf.enable_eager_execution()
np.random.seed(1000)

cnn_model = FCNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=1e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10,
                                           gamma=0.1)

# im = np.random.randint(255, size=(1, 5, 150, 150)).astype(np.float32)
# im_tensor = torch.tensor(im)
#
# output = cnn_model(im_tensor)
# print(output.shape)


batch_size = 32


train_dataset, train_dataset_size, valid_dataset, valid_dataset_size = \
    datasets_on_single_files_torch_segmentation(
        regions=[6], channels=[0, 1, 2, 3, 4],
        train_ratio=0.80, batch_size=batch_size)

train_on_preloaded_single_files_torch(
    cnn_model, train_dataset, train_dataset_size, valid_dataset, valid_dataset_size,
    folder=f"{data_path}/results/test_training_segmentation_on_6_torch",
    epochs=10,
    batch_size=batch_size,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=exp_lr_scheduler)

