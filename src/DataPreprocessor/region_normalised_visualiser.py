import logging
import pathlib
from typing import Tuple

import yaml
from PIL import Image

from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.DataPreprocessor.normalised_data import NormalisedData
from src.DataPreprocessor.preprocessed_data import PreprocessedData
from src.DataPreprocessor.raw_data_preprocessor import RawDataPreprocessor, FeatureValue
from src.DataPreprocessor.region_normaliser import RegionNormaliser
from src.config import data_path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class RegionNormalisedVisualiser:
    def __init__(self, region_id):
        self.gdal_backend = GdalBackend()
        with open(f"{data_path}/preprocessed/{region_id}/gdal_params.yaml", 'r') as stream:
            gdal_params = yaml.safe_load(stream)
        self.gdal_backend.set_params(gdal_params['driver_name'], gdal_params['projection'], eval(gdal_params['geotransform']))

        self.normalised_data = NormalisedData(region_id)
        self.normalised_data.load()
        self.visualisation_folder = f"../visualisation/{region_id}/"
        pathlib.Path(self.visualisation_folder).mkdir(exist_ok=True, parents=True)

    def plot_distributions(self):
        nrows, ncols = 3, 5
        f, axis = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 20))
        ax_ind = 0
        for key in self.normalised_data.channels.keys():
            if self.normalised_data.channels[key].ndim == 3:
                for d in range(self.normalised_data.channels[key].shape[2]):
                    ind_2d = np.unravel_index(ax_ind, (nrows, ncols))
                    cur_ax = axis[ind_2d[0], ind_2d[1]]
                    logging.info(f"plot normalised distribution for {key}_{d}")
                    sns.distplot(self.normalised_data.channels[key][:, :, d].flatten(), ax=cur_ax)
                    cur_ax.set_title(f'{key}_{d}')
                    ax_ind = ax_ind + 1
            elif self.normalised_data.channels[key].ndim == 2:
                ind_2d = np.unravel_index(ax_ind, (nrows, ncols))
                cur_ax = axis[ind_2d[0], ind_2d[1]]
                logging.info(f"plot normalised distribution for {key}")
                if np.count_nonzero(self.normalised_data.channels[key]) > 0:
                    sns.distplot(self.normalised_data.channels[key].flatten(), ax=cur_ax)
                cur_ax.set_title(f'{key}')
                ax_ind = ax_ind + 1
        plt.tight_layout()
        f.savefig(self.visualisation_folder + "normalised_features_distribution.png")
        f.clf()
        plt.close()

