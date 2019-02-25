from src.pipeline.predicting import predict

predict(models_folder="feature_erosion/trained_models_1", ensemble_size=1, classes=2, channels=[11], heatmap_mode="mean", stride=10, batch_size=100)