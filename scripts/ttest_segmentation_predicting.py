import sys
import torch
import numpy as np

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from src.DataPreprocessor.region_dataset import RegionDataset
from src.pipeline_torch.predicting import predict_torch

sys.path.extend(['../../EarthScienceFaultDetection'])

from src.LearningTorch.net_architecture import FCNet
from src.pipeline.predicting import predict, postprocess
from src.config import data_path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# cnn_model = FCNet()

folder = f"{data_path}/results/test_training_segmentation_on_6_torch"
training_output = torch.load(folder + '/model.pth', map_location=device)
cnn_model = training_output['model']

# predict_torch(datasets=[6], models_folder=f"{data_path}/results/test_training_on_6_torch", classes=2,
#         channels=[0, 1, 2, 3, 4], stride=50, batch_size=16)
# postprocess(datasets=[0], models_folder=f"{data_path}/results/test_training_on_6_torch", heatmap_mode="mean")


data_preprocessor = RegionDataset(6)

input_image = data_preprocessor.get_full_image()

sliced_input_image = input_image[4440:4590, 3528:3678, :]

input_data = np.expand_dims(
    sliced_input_image.astype(np.float32).transpose((2, 0, 1)),
    axis=0)

cnn_model.eval()
# model.requires_grad_(False)

prediction = cnn_model(torch.tensor(input_data))
