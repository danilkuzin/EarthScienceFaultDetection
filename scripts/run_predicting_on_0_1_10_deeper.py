import sys

from src.LearningKeras.net_architecture import cnn_four_layers

sys.path.extend(['../../EarthScienceFaultDetection'])

from src.pipeline.predicting import predict, postprocess

model = cnn_four_layers()

predict(datasets=[6, 7], model=model, models_folder='training_on_0_1_10_deeper', classes=2,
        channels=[0, 1, 2, 3, 4], stride=10, batch_size=16)
postprocess(datasets=[6, 7], models_folder='training_on_0_1_10_deeper', heatmap_mode="mean")

