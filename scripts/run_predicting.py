from src.pipeline.predicting import predict

predict(models_folder="trained_models_12", ensemble_size=2, classes=2, channels=[0, 1, 2])