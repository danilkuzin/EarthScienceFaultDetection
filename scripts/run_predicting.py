from src.pipeline.predicting import predict

predict(models_folder="feature_3_6_9_10_11/trained_models_12", ensemble_size=1, classes=2, channels=[3, 6, 10, 11], heatmap_mode="mean", stride=25, batch_size=100)