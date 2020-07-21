import sys
import numpy as np
import tensorflow as tf


sys.path.extend(['../../EarthScienceFaultDetection'])

from src.LearningTorch.net_architecture import cnn_150x150x5_torch
from src.pipeline_torch.training import datasets_on_single_files_torch, \
    train_on_preloaded_single_files_torch

from src.config import data_path

tf.enable_eager_execution()
np.random.seed(1000)

model, criterion, optimizer, exp_lr_scheduler = cnn_150x150x5_torch()

batch_size = 32

train_dataset, train_dataset_size, valid_dataset, valid_dataset_size = \
    datasets_on_single_files_torch(regions=[6], channels=[0, 1, 2, 3, 4],
                                   train_ratio=0.80, batch_size=batch_size)

train_on_preloaded_single_files_torch(
    model, train_dataset, train_dataset_size, valid_dataset, valid_dataset_size,
    folder=f"{data_path}/results/test_training_on_6_torch", epochs=10,
    batch_size=batch_size,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=exp_lr_scheduler)
