import io
import logging
import pathlib
from enum import Enum

import h5py
import numpy as np
import yaml

from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend


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
    def __init__(self, data_dir: pathlib.Path, data_io_backend: GdalBackend, max_shape=None):
        self.data_dir = data_dir
        self.channels = dict(elevation=None, slope=None, optical_rgb=None, nir=None, ultrablue=None, swir1=None, swir2=None,
                             panchromatic=None, curve=None, erosion=None)

        self.features = None
        self.data_io_backend = data_io_backend
        self.max_shape = max_shape

    def __check_crop_data(self):
        min_shape = self.get_data_shape()[0], self.get_data_shape()[1]
        for channel in self.channels.values():
            min_shape = min(channel.shape[0], min_shape[0]), min(channel.shape[1], min_shape[1])

        if self.max_shape:
            min_shape = min(min_shape[0], self.max_shape[0]), min(min_shape[1], self.max_shape[1])

        for ch_name, channel in self.channels.items():
            if channel.shape[0] > min_shape[0] or channel.shape[1] > min_shape[1]:
                logging.warning("{} images do not match in size".format(ch_name))
                self.channels[ch_name] = channel[:min_shape[0], :min_shape[1]]

    def __check_crop_features(self):
        if self.features.shape[0] > self.get_data_shape()[0] or self.features.shape[1] > self.get_data_shape()[1]:
            logging.warning("features do not match in size")
            self.features = self.features[:self.get_data_shape()[0], :self.get_data_shape()[1]]

    def load(self):
        logging.info('loading...')
        self.channels['elevation'] = self.data_io_backend.load_elevation(path=str(self.data_dir / 'elev.tif'))
        self.channels['slope'] = self.data_io_backend.load_slope(path=str(self.data_dir / 'slope.tif'))
        self.channels['optical_rgb'] = self.data_io_backend.load_optical(path_r=str(self.data_dir / 'r.tif'),
                                                             path_g=str(self.data_dir / 'g.tif'),
                                                             path_b=str(self.data_dir / 'b.tif'))
        try:
            self.channels['nir'] = self.data_io_backend.load_nir(path=str(self.data_dir / 'nir.tif'))
        except FileNotFoundError as err:
            self.channels['nir'] = np.zeros_like(self.channels['elevation'])
            logging.warning("Error: {}".format(err))

        try:
            self.channels['ultrablue'] = self.data_io_backend.load_ultrablue(path=str(self.data_dir / 'o.tif'))
        except FileNotFoundError as err:
            self.channels['ultrablue'] = np.zeros_like(self.channels['elevation'])
            logging.warning("Error: {}".format(err))

        try:
            self.channels['swir1'] = self.data_io_backend.load_swir1(path=str(self.data_dir / 'swir1.tif'))
        except FileNotFoundError as err:
            self.channels['swir1'] = np.zeros_like(self.channels['elevation'])
            logging.warning("Error: {}".format(err))

        try:
            self.channels['swir2'] = self.data_io_backend.load_swir2(path=str(self.data_dir / 'swir2.tif'))
        except FileNotFoundError as err:
            self.channels['swir2'] = np.zeros_like(self.channels['elevation'])
            logging.warning("Error: {}".format(err))

        try:
            self.channels['panchromatic'] = self.data_io_backend.load_panchromatic(path=str(self.data_dir / 'p.tif'))
        except FileNotFoundError as err:
            self.channels['panchromatic'] = np.zeros_like(self.channels['elevation'])
            logging.warning("Error: {}".format(err))

        try:
            self.channels['curve'] = self.data_io_backend.load_curve(path=str(self.data_dir / 'curv.tif'))
        except FileNotFoundError as err:
            self.channels['curve'] = np.zeros_like(self.channels['elevation'])
            logging.warning("Error: {}".format(err))

        try:
            self.channels['erosion'] = self.data_io_backend.load_erosion(path=str(self.data_dir / 'erode.tif'))
        except FileNotFoundError as err:
            self.channels['erosion'] = np.zeros_like(self.channels['elevation'])
            logging.warning("Error: {}".format(err))

        self.__check_crop_data()

        try:
            self.features = self.data_io_backend.load_features(path=str(self.data_dir / 'features.tif'))
        except FileNotFoundError:
            logging.warning("no features file presented, initialise with undefined")
            self.features = FeatureValue.UNDEFINED.value * np.ones_like(self.channels['elevation'])
        self.features = self.data_io_backend.append_additional_features(path=str(self.data_dir / 'additional_data/'), features=self.features)
        self.__check_crop_features()
        logging.info('loaded')

    def get_data_shape(self):
        return self.channels['optical_rgb'].shape[0], self.channels['optical_rgb'].shape[1], len(self.channels)

    def write_data(self, output_path: str):
        with h5py.File(f"{output_path}/data.h5", 'w') as hf:
            dict(elevation=None, slope=None, optical_rgb=None, nir=None, ultrablue=None, swir1=None, swir2=None,
                 panchromatic=None, curve=None, erosion=None)
            hf.create_dataset("elevation", data=self.channels['elevation'])
            hf.create_dataset("slope", data=self.channels['slope'])
            hf.create_dataset("optical_r", data=self.channels['optical_rgb'][0])
            hf.create_dataset("optical_g", data=self.channels['optical_rgb'][1])
            hf.create_dataset("optical_b", data=self.channels['optical_rgb'][2])
            hf.create_dataset("nir", data=self.channels['nir'])
            hf.create_dataset("ultrablue", data=self.channels['ultrablue'])
            hf.create_dataset("swir1", data=self.channels['swir1'])
            hf.create_dataset("swir2", data=self.channels['swir2'])
            hf.create_dataset("panchromatic", data=self.channels['panchromatic'])
            hf.create_dataset("curve", data=self.channels['curve'])
            hf.create_dataset("erosion", data=self.channels['erosion'])
            hf.create_dataset("features", data=self.features[1])

        with io.open(f"{output_path}/gdal_params.yaml", 'w', encoding='utf8') as outfile:
            yaml.dump(self.data_io_backend.get_params(), outfile, default_flow_style=False, allow_unicode=True)

