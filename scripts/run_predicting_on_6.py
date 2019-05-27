import sys
sys.path.extend(['../../EarthScienceFaultDetection'])

from src.pipeline.predicting import predict, postprocess

predict(datasets=[6, 7], models_folder='training_on_6', classes=2,
        channels=[0, 1, 2, 3, 4], stride=10, batch_size=16)
postprocess(datasets=[6, 7], models_folder='training_on_6', heatmap_mode="mean")