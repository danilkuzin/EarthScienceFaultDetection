from src.DataPreprocessor.DataIOBackend.backend import Backend
import cv2
import numpy as np


class OpencvBackend(Backend):
    def load_elevation(self, path: str) -> np.array:
        elevation = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if not elevation:
            raise FileNotFoundError(path)

    def load_slope(self, path: str) -> np.array:
        # todo check why it produces None image
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
