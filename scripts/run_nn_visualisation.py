import pathlib
import sys

from src.DataPreprocessor.region_dataset import RegionDataset, FeatureValue

sys.path.extend(['../../EarthScienceFaultDetection'])

from src.LearningKeras.net_architecture import cnn_150x150x5

from src.pipeline.nn_visualisation import NnVisualisation
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from src.config import data_path

np.random.seed(100)
tf.set_random_seed(100)

model=cnn_150x150x5()
model.load_weights(f"{data_path}/results/training_on_0_1_10_novel_elevation_normalisation/model.h5")

nn_visualiser = NnVisualisation(model=model, region_id=0, num_samples=10)
nn_visualiser.visualise_nn()