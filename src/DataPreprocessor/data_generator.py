from src.DataPreprocessor.data_preprocessor import DataPreprocessor
from typing import List, Tuple
import numpy as np


class DataGenerator:
    """
    A collection of data_preprocessors that feed data into training pipeline
    """
    def __init__(self, preprocessors: List[DataPreprocessor]):
        self.preprocessors = preprocessors

    def generator(self, batch_size: int, class_probabilities: np.array, patch_size: Tuple[int, int], channels: np.array):
        while True:
            img_batches = []
            lbl_batches = []
            for preprocessor in self.preprocessors:
                img_batch, lbl_batch = next(preprocessor.train_generator(
                    batch_size=batch_size,
                    class_probabilities=class_probabilities,
                    patch_size=patch_size,
                    channels=channels))
                img_batches.append(img_batch)
                lbl_batches.append(lbl_batch)
            yield img_batches, lbl_batches

