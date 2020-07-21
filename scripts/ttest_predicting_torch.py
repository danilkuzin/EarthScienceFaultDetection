import sys
import torch
import numpy as np

from src.DataPreprocessor.region_dataset import RegionDataset
from src.pipeline_torch.predicting import predict_torch

sys.path.extend(['../../EarthScienceFaultDetection'])

from src.LearningTorch.net_architecture import cnn_150x150x5_torch
from src.pipeline.predicting import predict, postprocess
from src.config import data_path

model, criterion, optimizer, exp_lr_scheduler = cnn_150x150x5_torch()

# predict_torch(datasets=[6], models_folder=f"{data_path}/results/test_training_on_6_torch", classes=2,
#         channels=[0, 1, 2, 3, 4], stride=50, batch_size=16)
# postprocess(datasets=[0], models_folder=f"{data_path}/results/test_training_on_6_torch", heatmap_mode="mean")


data_preprocessor = RegionDataset(0)

input_image = data_preprocessor.get_full_image()

sliced_input_image = input_image[:150, :150, :]

input_data = np.expand_dims(
    sliced_input_image.astype(np.float32).transpose((2, 0, 1)),
    axis=0)

model.eval()
# model.requires_grad_(False)

prediction = model(torch.tensor(input_data))
