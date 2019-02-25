from src.DataPreprocessor.data_preprocessor import DataPreprocessor, FeatureValue
import numpy as np
from PIL import Image

#todo this became too repetitive
class DataVisualiser:
    def __init__(self, preprocessor: DataPreprocessor):
        self.preprocessor = preprocessor

    def get_features_map_transparent(self, opacity):
        mask_rgba = np.zeros((self.preprocessor.features.shape[0], self.preprocessor.features.shape[1], 4), dtype=np.uint8)
        mask_rgba[np.where(self.preprocessor.features == FeatureValue.FAULT.value)] = [250, 0, 0, 0]
        mask_rgba[np.where(self.preprocessor.features == FeatureValue.FAULT_LOOKALIKE.value)] = [0, 250, 0, 0]
        mask_rgba[np.where(self.preprocessor.features == FeatureValue.NONFAULT.value)] = [0, 0, 250, 0]
        mask_rgba[:, :, 3] = opacity
        return Image.fromarray(mask_rgba)

    def get_optical_rgb(self):
        return Image.fromarray(self.preprocessor.channels['optical_rgb'])

    def get_optical_rgb_with_features_mask(self, opacity=60) -> Image:
        features_map = self.get_features_map_transparent(opacity)
        orig = self.get_optical_rgb().convert('RGBA')
        return Image.alpha_composite(orig, features_map)

    def get_elevation(self):
        orig = self.preprocessor.channels['elevation'].astype(np.float)
        orig = orig - np.min(orig)
        orig = orig * 255. / np.max(orig)
        return Image.fromarray(orig.astype(np.uint8))

    def get_elevation_with_features_mask(self, opacity=60):
        features_map = self.get_features_map_transparent(opacity)
        orig_im = self.get_elevation().convert('RGBA')
        return Image.alpha_composite(orig_im, features_map)

    def get_slope(self):
        orig = self.preprocessor.channels['slope'].astype(np.float)
        orig = orig - np.min(orig)
        orig = orig * 255. / np.max(orig)
        return Image.fromarray(orig.astype(np.uint8))

    def get_slope_with_features_mask(self, opacity=60):
        features_map = self.get_features_map_transparent(opacity)
        orig_im = self.get_slope().convert('RGBA')
        return Image.alpha_composite(orig_im, features_map)

    def get_nir(self):
        orig = self.preprocessor.channels['nir'].astype(np.float)
        orig = orig - np.min(orig)
        orig = orig * 255. / np.max(orig)
        return Image.fromarray(orig.astype(np.uint8))

    def get_nir_with_features_mask(self, opacity=60):
        features_map = self.get_features_map_transparent(opacity)
        orig_im = self.get_nir().convert('RGBA')
        return Image.alpha_composite(orig_im, features_map)

    def get_ir(self):
        orig = self.preprocessor.channels['ir'].astype(np.float)
        orig = orig - np.min(orig)
        orig = orig * 255. / np.max(orig)
        return Image.fromarray(orig.astype(np.uint8))

    def get_ir_with_features_mask(self, opacity=60):
        features_map = self.get_features_map_transparent(opacity)
        orig_im = self.get_ir().convert('RGBA')
        return Image.alpha_composite(orig_im, features_map)

    def get_swir1(self):
        orig = self.preprocessor.channels['swir1'].astype(np.float)
        orig = orig - np.min(orig)
        orig = orig * 255. / np.max(orig)
        return Image.fromarray(orig.astype(np.uint8))

    def get_swir1_with_features_mask(self, opacity=60):
        features_map = self.get_features_map_transparent(opacity)
        orig_im = self.get_swir1().convert('RGBA')
        return Image.alpha_composite(orig_im, features_map)

    def get_swir2(self):
        orig = self.preprocessor.channels['swir2'].astype(np.float)
        orig = orig - np.min(orig)
        orig = orig * 255. / np.max(orig)
        return Image.fromarray(orig.astype(np.uint8))

    def get_swir2_with_features_mask(self, opacity=60):
        features_map = self.get_features_map_transparent(opacity)
        orig_im = self.get_swir2().convert('RGBA')
        return Image.alpha_composite(orig_im, features_map)

    def get_panchromatic(self):
        orig = self.preprocessor.channels['panchromatic'].astype(np.float)
        orig = orig - np.min(orig)
        orig = orig * 255. / np.max(orig)
        return Image.fromarray(orig.astype(np.uint8))

    def get_panchromatic_with_features_mask(self, opacity=60):
        features_map = self.get_features_map_transparent(opacity)
        orig_im = self.get_panchromatic().convert('RGBA')
        return Image.alpha_composite(orig_im, features_map)

    def get_curve(self):
        orig = self.preprocessor.channels['curve'].astype(np.float)
        orig = orig - np.min(orig)
        orig = orig * 255. / np.max(orig)
        return Image.fromarray(orig.astype(np.uint8))

    def get_curve_with_features_mask(self, opacity=60):
        features_map = self.get_features_map_transparent(opacity)
        orig_im = self.get_curve().convert('RGBA')
        return Image.alpha_composite(orig_im, features_map)

    def get_erosion(self):
        orig = self.preprocessor.channels['erosion'].astype(np.float)
        orig = orig - np.min(orig)
        orig = orig * 255. / np.max(orig)
        return Image.fromarray(orig.astype(np.uint8))

    def get_erosion_with_features_mask(self, opacity=60):
        features_map = self.get_features_map_transparent(opacity)
        orig_im = self.get_erosion().convert('RGBA')
        return Image.alpha_composite(orig_im, features_map)
