from src.DataPreprocessor.data_preprocessor import DataPreprocessor
from typing import List

class DataGenerator:
    """
    A collection of data_preprocessors that feed data into training pipeline
    """
    def __init__(self, preprocessors: List[DataPreprocessor]):
        self.preprocessors = preprocessors

