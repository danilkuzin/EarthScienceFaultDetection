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
    def __init__(self, region_id, area_ind):
        self.region_id = region_id
        self.area_ind = area_ind
        self.preprocessed_data = None
        self.normalised_data = NormalisedData(region_id)

    def load(self):
        self.preprocessed_data = PreprocessedData(self.region_id)
        self.preprocessed_data.load()

    def normalise(self):
        optical_mean = 127.5
        optical_std = 255.
        optical_r_normalised = (self.preprocessed_data.channels['optical_rgb'][:, :, 0] - optical_mean) / optical_std
        optical_g_normalised = (self.preprocessed_data.channels['optical_rgb'][:, :, 1] - optical_mean) / optical_std
        optical_b_normalised = (self.preprocessed_data.channels['optical_rgb'][:, :, 2] - optical_mean) / optical_std
        self.normalised_data.channels['optical_rgb'] = np.stack((optical_r_normalised, optical_g_normalised, optical_b_normalised), axis=-1)

        input_path = f"{data_path}/normalised"
        with open(f"{input_path}/area_{self.area_ind}.yaml", 'r') as stream:
            features_areawide = yaml.safe_load(stream)

        elevation_mean, elevation_std = features_areawide['mean_elevation'], features_areawide['std_elevation']
        self.normalised_data.channels['elevation'] = (self.preprocessed_data.channels['elevation'] - elevation_mean) / elevation_std

        slope_mean = 45.
        slope_std = 90.
        self.normalised_data.channels['slope'] = (self.preprocessed_data.channels['slope'] - slope_mean) / slope_std
        self.normalised_data.features = self.preprocessed_data.features

    def save_results(self):
        self.normalised_data.save()




