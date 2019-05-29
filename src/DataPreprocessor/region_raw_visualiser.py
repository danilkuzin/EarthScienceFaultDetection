import logging
import pathlib
from typing import Tuple

import yaml
from PIL import Image

from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.DataPreprocessor.preprocessed_data import PreprocessedData
from src.DataPreprocessor.raw_data_preprocessor import RawDataPreprocessor, FeatureValue
from src.DataPreprocessor.region_normaliser import RegionNormaliser
from src.config import data_path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#todo consider extending PreprocessedData instead of using it as a property
class RegionRawVisualiser():
    def __init__(self, region_id):
        self.gdal_backend = GdalBackend()
        with open(f"{data_path}/preprocessed/{region_id}/gdal_params.yaml", 'r') as stream:
            gdal_params = yaml.safe_load(stream)
        self.gdal_backend.set_params(gdal_params['driver_name'], gdal_params['projection'], eval(gdal_params['geotransform']))

        self.preprocessed_data = PreprocessedData(region_id)
        self.preprocessed_data.load()
        self.visualisation_folder = f"../visualisation/{region_id}/"
        pathlib.Path(self.visualisation_folder).mkdir(exist_ok=True, parents=True)

    def write_features(self):
        mask_rgb = np.zeros((self.preprocessed_data.features.shape[0], self.preprocessed_data.features.shape[1], 3), dtype=np.uint8)
        mask_rgb[np.where(self.preprocessed_data.features == FeatureValue.FAULT.value)] = [250, 0, 0]
        mask_rgb[np.where(self.preprocessed_data.features == FeatureValue.FAULT_LOOKALIKE.value)] = [0, 250, 0]
        mask_rgb[np.where(self.preprocessed_data.features == FeatureValue.NONFAULT.value)] = [0, 0, 250]

        self.gdal_backend.write_image(self.visualisation_folder+"features", mask_rgb, crop=None)

    def get_channel_with_feature_mask(self, key: str, opacity: int) -> Image:
        features_map = self.get_features_map_transparent(opacity)
        orig_im = self.get_channel(key)
        if orig_im is not None:
            orig_im = orig_im.convert('RGBA')
            return Image.alpha_composite(orig_im, features_map)
        else:
            return None

    def write_channel_with_feature_mask(self, key: str, opacity: int, path: str, crop:Tuple):
        im = self.get_channel_with_feature_mask(key, opacity)
        if im is None:
            return
        im = np.array(im.convert('RGB'))
        self.gdal_backend.write_image(path, im, crop)

    def get_features_map_transparent(self, opacity: int):
        mask_rgba = np.zeros((self.preprocessed_data.features.shape[0], self.preprocessed_data.features.shape[1], 4),
                             dtype=np.uint8)
        mask_rgba[np.where(self.preprocessed_data.features == FeatureValue.FAULT.value)] = [250, 0, 0, 0]
        mask_rgba[np.where(self.preprocessed_data.features == FeatureValue.FAULT_LOOKALIKE.value)] = [0, 250, 0, 0]
        mask_rgba[np.where(self.preprocessed_data.features == FeatureValue.NONFAULT.value)] = [0, 0, 250, 0]
        mask_rgba[:, :, 3] = opacity
        return Image.fromarray(mask_rgba)

    def get_channel(self, key: str) -> Image:
        orig = self.preprocessed_data.channels[key]
        if orig is None:
            return None
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
        orig = self.preprocessed_data.channels[key]
        if orig is None:
            return None
        if orig.ndim == 3:
            self.gdal_backend.write_image(path, orig, crop)
        elif orig.ndim == 2:
            self.gdal_backend.write_surface(path, orig, crop)
        else:
            raise Exception()

    def plot_distributions(self):
        nrows, ncols = 3, 5
        f, axis = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 20))
        ax_ind = 0
        for key in self.preprocessed_data.channels.keys():
            if self.preprocessed_data.channels[key] is None:
                ax_ind = ax_ind + 1
                continue
            if self.preprocessed_data.channels[key].ndim == 3:
                for d in range(self.preprocessed_data.channels[key].shape[2]):
                    ind_2d = np.unravel_index(ax_ind, (nrows, ncols))
                    cur_ax = axis[ind_2d[0], ind_2d[1]]
                    logging.info(f"plot distribution for {key}_{d}")
                    sns.distplot(self.preprocessed_data.channels[key][:, :, d].flatten(), ax=cur_ax)
                    cur_ax.set_title(f'{key}_{d}')
                    ax_ind = ax_ind + 1
            elif self.preprocessed_data.channels[key].ndim == 2:
                ind_2d = np.unravel_index(ax_ind, (nrows, ncols))
                cur_ax = axis[ind_2d[0], ind_2d[1]]
                logging.info(f"plot distribution for {key}")
                if np.count_nonzero(self.preprocessed_data.channels[key]) > 0:
                    sns.distplot(self.preprocessed_data.channels[key].flatten(), ax=cur_ax)
                cur_ax.set_title(f'{key}')
                ax_ind = ax_ind + 1
        plt.tight_layout()
        f.savefig(self.visualisation_folder + "features_distribution.png")
        f.clf()
        plt.close()

    def plot_channels_with_features(self, crop):
        for key in self.preprocessed_data.channels.keys():
            self.write_channel_with_feature_mask(key=key, opacity=90, path=f"{self.visualisation_folder}features_{key}",
                                                            crop=crop)

    def plot_channels(self, crop):
        for key in self.preprocessed_data.channels.keys():
            self.write_channel(key=key, path=f"{self.visualisation_folder}{key}", crop=crop)