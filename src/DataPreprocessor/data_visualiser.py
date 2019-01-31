from src.DataPreprocessor.data_preprocessor import DataPreprocessor
import numpy as np
from PIL import Image


class DataVisualiser:
    def __init__(self, preprocessor: DataPreprocessor):
        self.preprocessor = preprocessor

    def get_features_map_transparent(self, opacity):
        mask_rgba = np.zeros((self.preprocessor.features.shape[0], self.preprocessor.features.shape[1], 4), dtype=np.uint8)
        mask_rgba[np.where(self.preprocessor.features == 1)] = [250, 0, 0, 0]
        mask_rgba[np.where(self.preprocessor.features == 2)] = [0, 250, 0, 0]
        mask_rgba[np.where(self.preprocessor.features == 3)] = [0, 0, 250, 0]
        mask_rgba[:, :, 3] = opacity
        return Image.fromarray(mask_rgba)

    def get_optical_rgb_with_features_mask(self, opacity=60):
        features_map = self.get_features_map_transparent(opacity)
        orig = Image.fromarray(self.preprocessor.optical_rgb).convert('RGBA')
        return Image.alpha_composite(orig, features_map)

    def get_elevation_with_features_mask(self, opacity=60):
        features_map = self.get_features_map_transparent(opacity)
        orig = Image.fromarray(self.preprocessor.elevation).convert('RGBA')
        return Image.alpha_composite(orig, features_map)