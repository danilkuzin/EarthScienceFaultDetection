from src.DataPreprocessor.data_preprocessor import DataPreprocessor
from typing import List
import numpy as np


class DataGenerator:
    """
    A collection of data_preprocessors that feed data into training pipeline
    """
    def __init__(self, preprocessors: List[DataPreprocessor]):
        self.preprocessors = preprocessors

    def get_sample(self, batch_size: int, class_probabilities: np.array, patch_size: List[int], channels: np.array):
        preprocessor_lbls = np.random.choice(len(self.preprocessors), batch_size)
        img_batch = np.zeros((batch_size,
                              patch_size[0],
                              patch_size[1],
                              channels.shape[0]))
        lbl_batch = np.zeros((batch_size, class_probabilities.shape[0]))

        for (preprocessor_ind, preprocessor) in enumerate(preprocessor_lbls):
            indices = np.where(preprocessor_lbls == preprocessor_ind)
            batch_size_for_preprocessor = indices.shape[0]
            img, labls = preprocessor.get_sample(batch_size=batch_size_for_preprocessor,
                                                 class_probabilities=class_probabilities,
                                                 patch_size=patch_size,
                                                 channels=channels)
            img_batch[indices] = img
            lbl_batch[indices] = labls
        return img_batch, lbl_batch, preprocessor_lbls

    def generator(self, batch_size: int, class_probabilities: np.array, patch_size: List[int], channels: np.array):
        while True:
            img_batch, lbl_batch, preprocessor_ind = self.get_sample(batch_size, class_probabilities, patch_size, channels)

            yield img_batch, lbl_batch

