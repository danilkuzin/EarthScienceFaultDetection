from PIL import Image

from src.DataPreprocessor.DataIOBackend.backend import DataIOBackend
import numpy as np


class PillowBackend(DataIOBackend):
    def __load_1d_raster(self, path: str) -> np.array:
        try:
            data = np.array(Image.open(path))
            return data
        except IOError:
            raise FileNotFoundError(path)

    def load_elevation(self, path: str) -> np.array:
        return self.__load_1d_raster(path)

    def load_slope(self, path: str) -> np.array:
        return self.__load_1d_raster(path)

    def load_optical(self, path_r: str, path_g: str, path_b: str) -> np.array:
        raise NotImplementedError("currently not supported")

    def load_features(self, path: str) -> np.array:
        try:
            features = np.array(Image.open(path)) - 1
            return features
        except IOError:
            raise FileNotFoundError(path)

    def load_nir(self, path: str) -> np.array:
        return self.__load_1d_raster(path)

    def load_ultrablue(self, path: str) -> np.array:
        return self.__load_1d_raster(path)

    def load_swir1(self, path: str) -> np.array:
        return self.__load_1d_raster(path)

    def load_swir2(self, path: str) -> np.array:
        return self.__load_1d_raster(path)

    def load_panchromatic(self, path: str) -> np.array:
        return self.__load_1d_raster(path)
