import sys
sys.path.extend(['../../EarthScienceFaultDetection'])

from src.pipeline.predicting import predict

predict(datasets=list(range(8)), models_folder="train_on_016_features_01234_no_additional/", ensemble_size=1, classes=2, channels=[0, 1, 2, 3, 4], heatmap_mode="mean", stride=50, batch_size=100)