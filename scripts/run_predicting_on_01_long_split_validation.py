import sys
sys.path.extend(['../../EarthScienceFaultDetection'])

from src.pipeline.predicting import predict

predict(datasets=list(range(8)), models_folder='training_on_01_long_split_validation', classes=2,
        channels=[0, 1, 2, 3, 4], heatmap_mode="mean", stride=10, batch_size=16)

