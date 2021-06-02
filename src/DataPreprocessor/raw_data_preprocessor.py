import io
import logging
import pathlib
from enum import Enum

import h5py
import numpy as np
import yaml

from src.DataPreprocessor.DataIOBackend.backend import DataIOBackend
from src.DataPreprocessor.preprocessed_data import PreprocessedData
from src.config import data_path

class OutOfBoundsException(Exception):
    pass


class Mode(Enum):
    TRAIN = 1
    TEST = 2


class FeatureValue(Enum):
    UNDEFINED = -1
    FAULT = 0
    FAULT_LOOKALIKE = 1
    NONFAULT = 2


class RawDataPreprocessor:
    def __init__(self, data_dir: pathlib.Path, data_io_backend: DataIOBackend, region_id: int, max_shape=None):
        self.data_dir = data_dir
        self.preprocessed_data = PreprocessedData(region_id)

        self.data_io_backend = data_io_backend
        self.max_shape = max_shape
        self.region_id = region_id

    def __check_crop_data(self):
        min_shape = self.get_data_shape()[0], self.get_data_shape()[1]
        for channel in self.preprocessed_data.channels.values():
            min_shape = min(channel.shape[0], min_shape[0]), min(channel.shape[1], min_shape[1])

        if self.max_shape:
            min_shape = min(min_shape[0], self.max_shape[0]), min(min_shape[1], self.max_shape[1])

        for ch_name, channel in self.preprocessed_data.channels.items():
            if channel.shape[0] > min_shape[0] or channel.shape[1] > min_shape[1]:
                logging.warning("{} images do not match in size".format(ch_name))
                self.preprocessed_data.channels[ch_name] = channel[:min_shape[0], :min_shape[1]]

    def __check_crop_features(self):
        if self.preprocessed_data.features.shape[0] > self.get_data_shape()[0] or self.preprocessed_data.features.shape[1] > self.get_data_shape()[1]:
            logging.warning("features do not match in size")
            self.preprocessed_data.features = self.preprocessed_data.features[:self.get_data_shape()[0], :self.get_data_shape()[1]]

    def load(self):
        logging.info('loading...')
        self.preprocessed_data.channels['elevation'] = self.data_io_backend.load_elevation(path=str(self.data_dir / 'elev.tif'))
        self.preprocessed_data.channels['slope'] = self.data_io_backend.load_slope(path=str(self.data_dir / 'slope.tif'))
        self.preprocessed_data.channels['optical_rgb'] = self.data_io_backend.load_optical(path_r=str(self.data_dir / 'r.tif'),
                                                             path_g=str(self.data_dir / 'g.tif'),
                                                             path_b=str(self.data_dir / 'b.tif'))
        try:
            self.preprocessed_data.channels['nir'] = self.data_io_backend.load_nir(path=str(self.data_dir / 'nir.tif'))
        except FileNotFoundError as err:
            self.preprocessed_data.channels['nir'] = np.zeros_like(self.preprocessed_data.channels['elevation'])
            logging.warning("Error: {}".format(err))

        try:
            self.preprocessed_data.channels['ultrablue'] = self.data_io_backend.load_ultrablue(path=str(self.data_dir / 'o.tif'))
        except FileNotFoundError as err:
            self.preprocessed_data.channels['ultrablue'] = np.zeros_like(self.preprocessed_data.channels['elevation'])
            logging.warning("Error: {}".format(err))

        try:
            self.preprocessed_data.channels['swir1'] = self.data_io_backend.load_swir1(path=str(self.data_dir / 'swir1.tif'))
        except FileNotFoundError as err:
            self.preprocessed_data.channels['swir1'] = np.zeros_like(self.preprocessed_data.channels['elevation'])
            logging.warning("Error: {}".format(err))

        try:
            self.preprocessed_data.channels['swir2'] = self.data_io_backend.load_swir2(path=str(self.data_dir / 'swir2.tif'))
        except FileNotFoundError as err:
            self.preprocessed_data.channels['swir2'] = np.zeros_like(self.preprocessed_data.channels['elevation'])
            logging.warning("Error: {}".format(err))

        try:
            self.preprocessed_data.channels['panchromatic'] = self.data_io_backend.load_panchromatic(path=str(self.data_dir / 'p.tif'))
        except FileNotFoundError as err:
            self.preprocessed_data.channels['panchromatic'] = np.zeros_like(self.preprocessed_data.channels['elevation'])
            logging.warning("Error: {}".format(err))

        try:
            self.preprocessed_data.channels['curve'] = self.data_io_backend.load_curve(path=str(self.data_dir / 'curv.tif'))
        except FileNotFoundError as err:
            self.preprocessed_data.channels['curve'] = np.zeros_like(self.preprocessed_data.channels['elevation'])
            logging.warning("Error: {}".format(err))

        try:
            self.preprocessed_data.channels['erosion'] = self.data_io_backend.load_erosion(path=str(self.data_dir / 'erode.tif'))
        except FileNotFoundError as err:
            self.preprocessed_data.channels['erosion'] = np.zeros_like(self.preprocessed_data.channels['elevation'])
            logging.warning("Error: {}".format(err))

        try:
            self.preprocessed_data.channels['topographic_roughness'] = self.data_io_backend.load_roughness(path=str(self.data_dir / 'TRI.tif'))
        except FileNotFoundError as err:
            self.preprocessed_data.channels['topographic_roughness'] = np.zeros_like(self.preprocessed_data.channels['elevation'])
            logging.warning("Error: {}".format(err))

        self.__check_crop_data()

        try:
            self.preprocessed_data.features = self.data_io_backend.load_features(path=str(self.data_dir / 'features.tif'))
        except FileNotFoundError:
            logging.warning("no features file presented, initialise with undefined")
            self.preprocessed_data.features = FeatureValue.UNDEFINED.value * np.ones_like(self.preprocessed_data.channels['elevation'])
        self.preprocessed_data.features = self.data_io_backend.append_additional_features(path=str(self.data_dir / 'additional_data/'), features=self.preprocessed_data.features)
        self.__check_crop_features()
        logging.info('loaded')

    def load_landsat(self):
        logging.info('loading...')
        self.preprocessed_data.channels['elevation'] = \
            self.data_io_backend.load_elevation(
                path=str(self.data_dir / 'elev_landsat.tif'))
        self.preprocessed_data.channels['optical_rgb'] = \
            self.data_io_backend.load_optical_landsat(
                path_r=str(self.data_dir / 'r_landsat.tif'),
                path_g=str(self.data_dir / 'g_landsat.tif'),
                path_b=str(self.data_dir / 'b_landsat.tif'))

        # try:
        #     self.preprocessed_data.channels['slope'] = \
        #         self.data_io_backend.load_nir_landsat(
        #             path=str(self.data_dir / 'slope_landsat.tif'))
        # except FileNotFoundError as err:
        #     self.preprocessed_data.channels['slope'] = np.zeros_like(
        #         self.preprocessed_data.channels['elevation'])
        #     logging.warning("Error: {}".format(err))

        try:
            self.preprocessed_data.channels['nir'] = \
                self.data_io_backend.load_nir_landsat(
                    path=str(self.data_dir / 'nir_landsat.tif'))
        except FileNotFoundError as err:
            self.preprocessed_data.channels['nir'] = np.zeros_like(
                self.preprocessed_data.channels['elevation'])
            logging.warning("Error: {}".format(err))

        try:
            self.preprocessed_data.channels['swir1'] = \
                self.data_io_backend.load_swir1_landsat(
                    path=str(self.data_dir / 'swir1_landsat.tif'))
        except FileNotFoundError as err:
            self.preprocessed_data.channels['swir1'] = np.zeros_like(
                self.preprocessed_data.channels['elevation'])
            logging.warning("Error: {}".format(err))

        try:
            self.preprocessed_data.channels['swir2'] = \
                self.data_io_backend.load_swir2_landsat(
                    path=str(self.data_dir / 'swir2_landsat.tif'))
        except FileNotFoundError as err:
            self.preprocessed_data.channels['swir2'] = np.zeros_like(
                self.preprocessed_data.channels['elevation'])
            logging.warning("Error: {}".format(err))

        try:
            self.preprocessed_data.channels['erosion'] = \
                self.data_io_backend.load_erosion(
                    path=str(self.data_dir / 'erode_landsat.tif'))
        except FileNotFoundError as err:
            self.preprocessed_data.channels['erosion'] = \
                np.zeros_like(self.preprocessed_data.channels['elevation'])
            logging.warning("Error: {}".format(err))

        try:
            self.preprocessed_data.channels['topographic_roughness'] = \
                self.data_io_backend.load_log_roughness(
                    path=str(self.data_dir / 'tri_landsat.tif'))
        except FileNotFoundError as err:
            self.preprocessed_data.channels['topographic_roughness'] = \
                np.zeros_like(self.preprocessed_data.channels['elevation'])
            logging.warning("Error: {}".format(err))

        try:
            self.preprocessed_data.channels['flow'] = \
                self.data_io_backend.load_log_flow(
                    path=str(self.data_dir / 'flow_log.tif'))
        except FileNotFoundError as err:
            self.preprocessed_data.channels['flow'] = \
                np.zeros_like(self.preprocessed_data.channels['elevation'])
            logging.warning("Error: {}".format(err))

        try:
            self.preprocessed_data.channels['sar1'] = \
                self.data_io_backend.load_sar1(
                    path=str(self.data_dir / 'sar1_landsat.tif'))
        except FileNotFoundError as err:
            self.preprocessed_data.channels['sar1'] = \
                np.zeros_like(self.preprocessed_data.channels['elevation'])
            logging.warning("Error: {}".format(err))

        try:
            self.preprocessed_data.channels['sar2'] = \
                self.data_io_backend.load_sar2(
                    path=str(self.data_dir / 'sar2_landsat.tif'))
        except FileNotFoundError as err:
            self.preprocessed_data.channels['sar2'] = \
                np.zeros_like(self.preprocessed_data.channels['elevation'])
            logging.warning("Error: {}".format(err))

        try:
            self.preprocessed_data.channels['incision'] = \
                self.data_io_backend.load_incision(
                    path=str(self.data_dir / 'incision_landsat.tif'))
        except FileNotFoundError as err:
            self.preprocessed_data.channels['incision'] = \
                np.zeros_like(self.preprocessed_data.channels['elevation'])
            logging.warning("Error: {}".format(err))

        # self.__check_crop_data()

        try:
            self.preprocessed_data.features = \
                self.data_io_backend.load_features(
                    path=str(self.data_dir / 'features.tif'))
        except FileNotFoundError:
            logging.warning(
                "no features file presented, initialise with undefined")
            self.preprocessed_data.features = \
                FeatureValue.UNDEFINED.value * \
                np.ones_like(self.preprocessed_data.channels['elevation'])
        self.preprocessed_data.features = \
            self.data_io_backend.append_additional_features(
                path=str(self.data_dir / 'additional_data/'),
                features=self.preprocessed_data.features)
        # self.__check_crop_features()
        logging.info('loaded')

    def get_data_shape(self):
        return self.preprocessed_data.channels['optical_rgb'].shape[0], \
               self.preprocessed_data.channels['optical_rgb'].shape[1], \
               len(self.preprocessed_data.channels)

    def write_data(self):
        self.preprocessed_data.save()
        output_path = f"{data_path}/preprocessed/{self.region_id}"
        with io.open(f"{output_path}/gdal_params.yaml", 'w', encoding='utf8') as outfile:
            yaml.dump(self.data_io_backend.get_params(), outfile, default_flow_style=False, allow_unicode=True)

    def write_data_landsat(self):
        self.preprocessed_data.save_landsat()
        output_path = f"{data_path}/preprocessed/{self.region_id}"
        with io.open(f"{output_path}/gdal_params.yaml", 'w', encoding='utf8') as outfile:
            yaml.dump(self.data_io_backend.get_params(), outfile, default_flow_style=False, allow_unicode=True)

