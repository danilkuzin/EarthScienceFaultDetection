from src.DataPreprocessor.data_preprocessor import DataPreprocessor
from typing import List
import numpy as np

class DataGenerator:
    """
    A collection of data_preprocessors that feed data into training pipeline
    """
    def __init__(self, preprocessors: List[DataPreprocessor]):
        self.preprocessors = preprocessors

    def generator(self, batch_size, class_probabilities, patch_size, channels):
        preprocessor_lbls = np.random.choice(len(self.preprocessors), batch_size)
        for (preprocessor_ind, preprocessor) in enumerate(preprocessor_lbls):
            np.sum(preprocessor_lbls == preprocessor_ind)
            preprocessor.get_sample()

