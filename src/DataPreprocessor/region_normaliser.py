import h5py
import yaml
from PIL import Image

from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.DataPreprocessor.normalised_data import NormalisedData
from src.DataPreprocessor.preprocessed_data import PreprocessedData
from src.DataPreprocessor.raw_data_preprocessor import FeatureValue
from src.config import data_preprocessor_params, areas, data_path
import numpy as np

import pathlib
import io


class RegionNormaliser:
    optical_mean = 127.5
    optical_std = 255.
    slope_mean = 45.
    slope_std = 90.
    roughness_mean = 50.
    roughness_std = 100.

    def __init__(self, region_id, area_ind):
        self.region_id = region_id
        self.area_ind = area_ind
        self.preprocessed_data = None
        self.normalised_data = NormalisedData(region_id)

    def load(self):
        self.preprocessed_data = PreprocessedData(self.region_id)
        self.preprocessed_data.load()

    def normalise(self):

        optical_r_normalised = (self.preprocessed_data.channels['optical_rgb'][:, :, 0] - self.optical_mean) / self.optical_std
        optical_g_normalised = (self.preprocessed_data.channels['optical_rgb'][:, :, 1] - self.optical_mean) / self.optical_std
        optical_b_normalised = (self.preprocessed_data.channels['optical_rgb'][:, :, 2] - self.optical_mean) / self.optical_std
        self.normalised_data.channels['optical_rgb'] = np.stack((optical_r_normalised, optical_g_normalised, optical_b_normalised), axis=-1)

        input_path = f"{data_path}/normalised"
        with open(f"{input_path}/area_{self.area_ind}.yaml", 'r') as stream:
            features_areawide = yaml.safe_load(stream)

        elevation_mean, elevation_std = features_areawide['mean_elevation'], features_areawide['std_elevation']
        self.normalised_data.channels['elevation'] = (self.preprocessed_data.channels['elevation'] - elevation_mean) / elevation_std

        self.normalised_data.channels['slope'] = (self.preprocessed_data.channels['slope'] - self.slope_mean) / self.slope_std

        self.normalised_data.channels['nir'] = (self.preprocessed_data.channels['nir'] - self.optical_mean) / self.optical_std

        self.normalised_data.channels['topographic_roughness'] = \
            (self.preprocessed_data.channels['topographic_roughness'] -
             self.roughness_mean) / self.roughness_std

        self.normalised_data.features = self.preprocessed_data.features

    def denormalise_patch(self, patch):
        normalised_optical_r = patch[:, :, 0]
        normalised_optical_g = patch[:, :, 1]
        normalised_optical_b = patch[:, :, 2]
        normalised_elevation = patch[:, :, 3]
        normalised_slope = patch[:, :, 4]

        optical_r = normalised_optical_r * self.optical_std + self.optical_mean
        optical_g = normalised_optical_g * self.optical_std + self.optical_mean
        optical_b = normalised_optical_b * self.optical_std + self.optical_mean

        input_path = f"{data_path}/normalised"
        with open(f"{input_path}/area_{self.area_ind}.yaml", 'r') as stream:
            features_areawide = yaml.safe_load(stream)

        elevation_mean, elevation_std = features_areawide['mean_elevation'], features_areawide['std_elevation']
        elevation = normalised_elevation * elevation_std + elevation_mean

        slope = normalised_slope * self.slope_std + self.slope_mean

        return np.stack((optical_r, optical_g, optical_b, elevation, slope), axis=-1)

    def save_results(self):
        self.normalised_data.save()




