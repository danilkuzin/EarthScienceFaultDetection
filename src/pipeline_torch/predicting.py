from typing import List

import h5py
import numpy as np
import tensorflow as tf
import torch
import yaml

from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.DataPreprocessor.region_dataset import RegionDataset
from src.LearningKeras.train import KerasTrainer
from src.LearningTorch.train import TorchTrainer
from src.postprocessing.postprocessor import PostProcessor

from src.config import data_path

np.random.seed(1)
tf.set_random_seed(2)


def predict_torch(datasets: List[int], models_folder, classes, channels, stride=25, batch_size=20):
    trainer = TorchTrainer()
    trainer.load(input_path=models_folder)

    for ind in datasets:
        data_preprocessor = RegionDataset(ind)

        boxes, probs = trainer.apply_for_sliding_window(
            data_preprocessor=data_preprocessor, patch_size=(150, 150), stride=stride, batch_size=batch_size,
            channels=channels)

        with h5py.File(f'{models_folder}/sliding_window_{ind}.h5', 'w') as hf:
            hf.create_dataset("boxes", data=boxes)
            hf.create_dataset("probs", data=probs)




