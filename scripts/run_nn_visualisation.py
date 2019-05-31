import pathlib
import sys

from src.DataPreprocessor.region_dataset import RegionDataset, FeatureValue

sys.path.extend(['../../EarthScienceFaultDetection'])

from src.LearningKeras.net_architecture import cnn_150x150x5

from src.pipeline.nn_visualisation import NnVisualisation
import numpy as np

import matplotlib.pyplot as plt

model=cnn_150x150x5()
model.load_weights('updated_heatmaps_trained_on_6/model.h5')

nn_visualiser = NnVisualisation(model=model, region_id=6, num_samples=6)
nn_visualiser.visualise_nn()