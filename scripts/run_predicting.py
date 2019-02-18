from src.pipeline.predicting import predict

predict(models_folder="2class_training_trained_models_12", ensemble_size=2, classes=2, channels=[0, 1, 2, 3, 4], heatmap_mode="mean")