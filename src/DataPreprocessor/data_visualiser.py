from typing import Tuple

from src.DataPreprocessor.data_preprocessor import DataPreprocessor, FeatureValue
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src.DataPreprocessor.region_dataset import RegionDataset


class DataVisualiser:
    def __init__(self, preprocessor: RegionDataset):
        self.preprocessor = preprocessor

    def get_features_map_transparent(self, opacity: int):
        mask_rgba = np.zeros((self.preprocessor.features.shape[0], self.preprocessor.features.shape[1], 4),
                             dtype=np.uint8)
        mask_rgba[np.where(self.preprocessor.features == FeatureValue.FAULT.value)] = [250, 0, 0, 0]
        mask_rgba[np.where(self.preprocessor.features == FeatureValue.FAULT_LOOKALIKE.value)] = [0, 250, 0, 0]
        mask_rgba[np.where(self.preprocessor.features == FeatureValue.NONFAULT.value)] = [0, 0, 250, 0]
        mask_rgba[:, :, 3] = opacity
        return Image.fromarray(mask_rgba)

    def get_channel(self, key: str) -> Image:
        orig = self.preprocessor.channels[key]
        if orig.ndim == 3:
            return Image.fromarray(orig)
        elif orig.ndim == 2:
            orig = orig.astype(np.float)
            orig = orig - np.min(orig)
            orig = orig * 255. / np.max(orig)
            return Image.fromarray(orig.astype(np.uint8))
        else:
            raise Exception()

    def write_channel(self, key: str, path: str, crop:Tuple):
        orig = self.preprocessor.channels[key]
        if orig.ndim == 3:
            self.preprocessor.data_io_backend.write_image(path, orig, crop)
        elif orig.ndim == 2:
            self.preprocessor.data_io_backend.write_surface(path, orig, crop)
        else:
            raise Exception()

    def write_features(self, path: str):
        mask_rgb = np.zeros((self.preprocessor.features.shape[0], self.preprocessor.features.shape[1], 3),
                             dtype=np.uint8)
        mask_rgb[np.where(self.preprocessor.features == FeatureValue.FAULT.value)] = [250, 0, 0]
        mask_rgb[np.where(self.preprocessor.features == FeatureValue.FAULT_LOOKALIKE.value)] = [0, 250, 0]
        mask_rgb[np.where(self.preprocessor.features == FeatureValue.NONFAULT.value)] = [0, 0, 250]

        self.preprocessor.data_io_backend.write_image(path, mask_rgb, crop=None)

    def get_channel_with_feature_mask(self, key: str, opacity: int) -> Image:
        features_map = self.get_features_map_transparent(opacity)
        orig_im = self.get_channel(key).convert('RGBA')
        return Image.alpha_composite(orig_im, features_map)

    def write_channel_with_feature_mask(self, key: str, opacity: int, path: str, crop:Tuple):
        im = self.get_channel_with_feature_mask(key, opacity)
        im = np.array(im.convert('RGB'))
        self.preprocessor.data_io_backend.write_image(path, im, crop)
