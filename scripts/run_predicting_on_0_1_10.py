import sys

from src.LearningKeras.net_architecture import cnn_150x150x5

sys.path.extend(['../../EarthScienceFaultDetection'])

from src.pipeline.predicting import predict, postprocess
from src.config import data_path

model = cnn_150x150x5()

predict(datasets=[2, 11], model=model, models_folder=f"{data_path}/results/training_on_0_1_10_novel_elevation_normalisation", classes=2,
        channels=[0, 1, 2, 3, 4], stride=5, batch_size=16)
postprocess(datasets=[2, 11], models_folder=f"{data_path}/results/training_on_0_1_10_novel_elevation_normalisation", heatmap_mode="mean")

