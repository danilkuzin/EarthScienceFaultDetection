import sys
sys.path.extend(['../../EarthScienceFaultDetection'])

from src.pipeline.predicting import predict, postprocess
predict(datasets=list(range(8)), models_folder='training_on_01_long_split_validation', classes=2,
        channels=[0, 1, 2, 3, 4], stride=10, batch_size=16)
postprocess(datasets=list(range(8)), models_folder='training_on_01_long_split_validation', heatmap_mode="mean")

