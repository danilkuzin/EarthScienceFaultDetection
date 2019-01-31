from PIL import Image

from src.DataPreprocessor.Backend.backend import Backend
import numpy as np


class PillowBackend(Backend):
    def load_elevation(self, path: str) -> np.array:
        try:
            elevation = np.array(Image.open(path))
            return elevation
        except IOError:
            raise FileNotFoundError(path)

    def load_slope(self, path: str) -> np.array:
        try:
            slope = np.array(Image.open(path))
            return slope
        except IOError:
            raise FileNotFoundError(path)

    def load_optical(self, path_r: str, path_g: str, path_b: str) -> np.array:
        # todo check this
        raise NotImplementedError("currently not supported")

    def load_features(self, path: str) -> np.array:
        try:
            features = np.array(Image.open(path))
            return features
        except IOError:
            raise FileNotFoundError(path)
