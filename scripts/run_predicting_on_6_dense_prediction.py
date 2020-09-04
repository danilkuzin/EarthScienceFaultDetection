import sys

from src.LearningKeras.net_architecture import cnn_150x150x5_fully_conv

sys.path.extend(['../../EarthScienceFaultDetection'])

from src.pipeline.predicting import predict, postprocess
from src.config import data_path

model = cnn_150x150x5_fully_conv()

predict_fully_conv(datasets=[6], models_folder=f"{data_path}/results/training_on_6_only_manual_faults_novel_elevation_normalisation_reduced_nonfault_area", classes=2,
        channels=[0, 1, 2, 3, 4], stride=15, batch_size=16, model=model)
postprocess_fully_conv(datasets=[6], models_folder=f"{data_path}/results/training_on_6_only_manual_faults_novel_elevation_normalisation_reduced_nonfault_area", heatmap_mode="mean")
