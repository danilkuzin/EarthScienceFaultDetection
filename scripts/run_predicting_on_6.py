import sys

from src.LearningKeras.net_architecture import cnn_150x150x5

sys.path.extend(['../../EarthScienceFaultDetection'])

from src.pipeline.predicting import predict, postprocess
from src.config import data_path

model = cnn_150x150x5()

predict(datasets=[6, 7], models_folder=f"{data_path}/results/training_on_6_only_manual_faults_novel_elevation_normalisation", classes=2,
        channels=[0, 1, 2, 3, 4], stride=10, batch_size=16, model=model)
postprocess(datasets=[6, 7], models_folder=f"{data_path}/results/training_on_6_only_manual_faults_novel_elevation_normalisation", heatmap_mode="mean")