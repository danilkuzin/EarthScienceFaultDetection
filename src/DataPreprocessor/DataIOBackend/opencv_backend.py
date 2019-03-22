from src.DataPreprocessor.DataIOBackend.backend import DataIOBackend
import cv2
import numpy as np


class OpencvBackend(DataIOBackend):
    def load_elevation(self, path: str) -> np.array:
        elevation = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if not elevation:
            raise FileNotFoundError(path)
        return elevation

    def load_slope(self, path: str) -> np.array:
        cv2.imread(path, cv2.IMREAD_LOAD_GDAL)
        raise NotImplementedError("currently not supported")

    def load_optical(self, path_r: str, path_g: str, path_b: str) -> np.array:
        optical_r = cv2.imread(path_r)[:, :, 0]
        optical_g = cv2.imread(path_g)[:, :, 0]
        optical_b = cv2.imread(path_b)[:, :, 0]
        if not any([optical_r, optical_g, optical_b]):
            raise FileNotFoundError([path_r, path_g, path_b])
        return np.dstack((optical_r, optical_g, optical_b))

    def load_features(self, path: str) -> np.array:
        features = cv2.imread(path, cv2.IMREAD_UNCHANGED) - 1
        if not features:
            raise FileNotFoundError(path)
        return features

    def load_nir(self, path: str) -> np.array:
        nir = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if not nir:
            raise FileNotFoundError(path)
        return nir

    def load_ultrablue(self, path: str) -> np.array:
        ir = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if not ir:
            raise FileNotFoundError(path)
        return ir

    def load_swir1(self, path: str) -> np.array:
        ir = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if not ir:
            raise FileNotFoundError(path)
        return ir

    def load_swir2(self, path: str) -> np.array:
        ir = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if not ir:
            raise FileNotFoundError(path)
        return ir

    def load_panchromatic(self, path: str) -> np.array:
        ir = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if not ir:
            raise FileNotFoundError(path)
        return ir


