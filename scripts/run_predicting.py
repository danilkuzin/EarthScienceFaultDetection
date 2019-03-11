from src.pipeline.predicting import predict

predict(models_folder="new_labels_all_nonfaults/trained_models_01", ensemble_size=1, classes=2, channels=[0, 1, 2, 3, 4], heatmap_mode="mean", stride=25, batch_size=100)