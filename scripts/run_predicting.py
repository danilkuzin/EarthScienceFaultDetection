from src.pipeline.predicting import predict

predict(datasets=list(range(6)), models_folder="train_on_6_only_longer/trained_models_6", ensemble_size=1, classes=2, channels=[0, 1, 2, 3, 4], heatmap_mode="mean", stride=50, batch_size=100)