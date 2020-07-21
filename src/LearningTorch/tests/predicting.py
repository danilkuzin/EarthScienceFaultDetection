import sys

from src.pipeline_torch.predicting import predict_torch

sys.path.extend(['../../EarthScienceFaultDetection'])

from src.LearningTorch.net_architecture import cnn_150x150x5_torch
from src.pipeline.predicting import predict, postprocess
from src.config import data_path

model = cnn_150x150x5_torch()

predict_torch(datasets=[6], models_folder=f"../../{data_path}/results/test_training_on_6_torch", classes=2,
        channels=[0, 1, 2, 3, 4], stride=50, batch_size=16)
postprocess(datasets=[6], models_folder=f"../../{data_path}/results/test_training_on_6_torch", heatmap_mode="mean")
