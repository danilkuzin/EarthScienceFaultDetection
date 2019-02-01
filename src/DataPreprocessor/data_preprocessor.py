from tempfile import NamedTemporaryFile

import itertools
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt

import logging
import os
import pathlib
#import gdal
from osgeo import gdal_array, osr
import struct
import tensorflow as tf

from PIL import Image

#TODO rewrite this as some tf.Dataset.from_generator or keras.ImageDataGenerator that feeds data in the same manner
from tqdm import trange, tqdm

from src.DataPreprocessor.DataIOBackend.backend import Backend
from src.DataPreprocessor.PatchesOutputBackend.backend import PatchesOutputBackend


class OutOfBoundsException(Exception):
    pass


class GdalFileException(Exception):
    pass

class Mode(Enum):
    TRAIN = 1
    TEST = 2

class FeatureValue(Enum):
    UNDEFINED = 0
    FAULT = 1
    FAULT_LOOKALIKE = 2
    NONFAULT = 3

class DatasetType(Enum):
    TRAIN = 1,
    VALIDATION = 2,
    TEST = 3

# todo include lookalikes as well
# todo add random seed
# todo move each enum into corresponding class with corresponding functions (load im, write dataset, etc)
# todo class is too big, consider separating classes
class DataPreprocessor:
    def __init__(self, data_dir: str, backend: Backend, filename_prefix: str, mode, seed: int):
        np.random.seed(seed)
        self.data_dir = data_dir
        self.elevation = None
        self.slope = None
        self.optical_rgb = None
        self.nir = None
        self.ir = None
        self.swir1 = None
        self.swir2 = None
        self.panchromatic = None
        self.features = None
        # todo move patch_size to sampling parameters in function
        self.patch_size = (150, 150)
        self.center_size = (50, 50)
        self.dirs = dict()
        self.datasets_sizes = dict()
        self.gdal_options = dict()
        self.datasets_sizes[DatasetType.TRAIN.name + "_" + FeatureValue.FAULT.name] = 100
        self.datasets_sizes[DatasetType.TRAIN.name + "_" + FeatureValue.NONFAULT.name] = 100
        self.datasets_sizes[DatasetType.VALIDATION.name + "_" + FeatureValue.FAULT.name] = 20
        self.datasets_sizes[DatasetType.VALIDATION.name + "_" + FeatureValue.NONFAULT.name] = 20
        self.datasets_sizes[DatasetType.TEST.name + "_" + FeatureValue.FAULT.name] = 10
        self.datasets_sizes[DatasetType.TEST.name + "_" + FeatureValue.NONFAULT.name] = 10
        self.num_channels = 5 # r, g, b, elevation, slope
        self.true_test_classes = None
        self.filename_prefix = filename_prefix
        self.mode = mode
        self.normalised = False
        self.prepare_folders()
        self.load(backend)

    def prepare_folders(self):
        if self.mode == Mode.TRAIN:
            self.dirs['train_fault'] = self.data_dir + "learn/train/fault/"
            self.dirs['train_nonfault'] = self.data_dir + "learn/train/nonfault/"
            self.dirs['valid_fault'] = self.data_dir + "learn/valid/fault/"
            self.dirs['valid_nonfault'] = self.data_dir + "learn/valid/nonfault/"
            self.dirs['test_w_labels_fault'] = self.data_dir + "learn/test_with_labels/fault/"
            self.dirs['test_w_labels_nonfault'] = self.data_dir + "learn/test_with_labels/nonfault/"
            self.dirs['test'] = self.data_dir + "learn/test/test/"
            pathlib.Path(self.dirs['train_fault']).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.dirs['train_nonfault']).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.dirs['valid_fault']).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.dirs['valid_nonfault']).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.dirs['test_w_labels_fault']).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.dirs['test_w_labels_nonfault']).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.dirs['test']).mkdir(parents=True, exist_ok=True)
        self.dirs['all_patches'] = self.data_dir + "all/"
        pathlib.Path(self.dirs['all_patches']).mkdir(parents=True, exist_ok=True)

    def load(self, backend):
        logging.info('loading...')
        self.elevation = backend.load_elevation(path=self.data_dir + self.filename_prefix + '_elev.tif')
        self.slope = backend.load_slope(path=self.data_dir + self.filename_prefix + '_slope.tif')
        self.optical_rgb = backend.load_optical(path_r=self.data_dir + self.filename_prefix + '_R.tif',
                                                path_g=self.data_dir + self.filename_prefix + '_G.tif',
                                                path_b=self.data_dir + self.filename_prefix + '_B.tif')
        logging.warning("optical images are not match in 1-2 pixels in size")
        self.optical_rgb = self.optical_rgb[:self.elevation.shape[0], :self.elevation.shape[1]]
        plt.imsave(self.data_dir+'data.tif', self.optical_rgb)
        if self.mode == Mode.TRAIN:
            self.features = backend.load_features(path=self.data_dir+'feature_categories.tif')
        logging.info('loaded')

    def borders_from_center(self, center):
        left_border = center[0] - self.patch_size[0] // 2
        right_border = center[0] + self.patch_size[0] // 2
        top_border = center[1] - self.patch_size[1] // 2
        bottom_border = center[1] + self.patch_size[1] // 2

        if not (0 < left_border < self.optical_rgb.shape[0]
                and 0 < right_border < self.optical_rgb.shape[0]
                and 0 < top_border < self.optical_rgb.shape[1]
                and 0 < bottom_border < self.optical_rgb.shape[1]):
            raise OutOfBoundsException

        return left_border, right_border, top_border, bottom_border

    def concatenate_full_patch(self, left_border: int, right_border: int, top_border: int, bottom_border: int):
        return np.concatenate((self.optical_rgb[left_border:right_border, top_border:bottom_border],
                               np.expand_dims(self.elevation[left_border:right_border, top_border:bottom_border], axis=2),
                               np.expand_dims(self.slope[left_border:right_border, top_border:bottom_border], axis=2)),
                              axis=2)

    def sample_fault_patch(self):
        """if an image patch contains fault bit in the center area than assign it as a fault - go through fault lines
        and sample patches"""
        fault_locations = np.argwhere(self.features == FeatureValue.FAULT.value)
        sampled = False
        left_border, right_border, top_border, bottom_border = None, None, None, None
        while not sampled:
            samples_ind = np.random.randint(fault_locations.shape[0])
            try:
                left_border, right_border, top_border, bottom_border = self.borders_from_center(
                    fault_locations[samples_ind])
                logging.info(
                    "extracting patch {}:{}, {}:{}".format(left_border, right_border, top_border, bottom_border))
                sampled = True
            except OutOfBoundsException:
                sampled = False

        return self.concatenate_full_patch(left_border, right_border, top_border, bottom_border)

    def sample_fault_lookalike_patch(self):
        """if an image patch contains fault lookalike bit in the center area than assign it as a fault - go through
        fault lookalike lines and sample patches"""
        fault_locations = np.argwhere(self.features == FeatureValue.FAULT_LOOKALIKE.value)
        sampled = False
        left_border, right_border, top_border, bottom_border = None, None, None, None
        while not sampled:
            samples_ind = np.random.randint(fault_locations.shape[0])
            try:
                left_border, right_border, top_border, bottom_border = self.borders_from_center(
                    fault_locations[samples_ind])
                logging.info(
                    "extracting patch {}:{}, {}:{}".format(left_border, right_border, top_border, bottom_border))
                sampled = True
            except OutOfBoundsException:
                sampled = False

        return self.concatenate_full_patch(left_border, right_border, top_border, bottom_border)

    def sample_nonfault_patch(self):
        """if an image path contains only nonfault bits, than assign it as a non-fault"""
        nonfault_locations = np.argwhere(self.features == FeatureValue.NONFAULT.value)
        sampled = False
        left_border, right_border, top_border, bottom_border = None, None, None, None
        while not sampled:
            samples_ind = np.random.randint(nonfault_locations.shape[0])
            try:
                left_border, right_border, top_border, bottom_border = self.borders_from_center(
                    nonfault_locations[samples_ind])
                logging.info(
                    "trying patch {}:{}, {}:{} as nonfault".format(left_border, right_border, top_border, bottom_border))
                is_probably_fault = False
                for i1, i2 in itertools.product(range(self.patch_size[0]), range(self.patch_size[1])):
                    if self.features[left_border + i1][top_border + i2] != FeatureValue.NONFAULT.value:
                        is_probably_fault = True
                        logging.info("probably fault")
                        break
                if not is_probably_fault:
                    logging.info("nonfault")
                    sampled = True
            except OutOfBoundsException:
                sampled = False
        return self.concatenate_full_patch(left_border, right_border, top_border, bottom_border)

    def sample_patch(self, label):
        if label==FeatureValue.FAULT:
            return self.sample_fault_patch()
        if label==FeatureValue.FAULT_LOOKALIKE:
            return self.sample_fault_lookalike_patch()
        elif label==FeatureValue.NONFAULT:
            return self.sample_nonfault_patch()

    def prepare_datasets(self, output_backend):
        self.prepare_dataset(output_backend, DatasetType.TRAIN, FeatureValue.FAULT)
        self.prepare_dataset(output_backend, DatasetType.TRAIN, FeatureValue.NONFAULT)
        self.prepare_dataset(output_backend, DatasetType.TRAIN, FeatureValue.FAULT_LOOKALIKE)
        self.prepare_dataset(output_backend, DatasetType.VALIDATION, FeatureValue.FAULT)
        self.prepare_dataset(output_backend, DatasetType.VALIDATION, FeatureValue.NONFAULT)
        self.prepare_dataset(output_backend, DatasetType.VALIDATION, FeatureValue.FAULT_LOOKALIKE)
        self.prepare_dataset(output_backend, DatasetType.TEST, FeatureValue.FAULT)
        self.prepare_dataset(output_backend, DatasetType.TEST, FeatureValue.NONFAULT)
        self.prepare_dataset(output_backend, DatasetType.TEST, FeatureValue.FAULT_LOOKALIKE)

    def prepare_dataset(self, output_backend: PatchesOutputBackend, data_type, label):
        category = data_type.name + "_" + label.name
        arr = np.zeros(
            (self.datasets_sizes[category], self.patch_size[0], self.patch_size[1], self.num_channels))
        for i in trange(self.datasets_sizes[category]):
            arr[i] = self.sample_patch(label)
        output_backend.save(arr, label==1 if 0 else 1, self.dirs[category])

    def prepare_all_patches(self, backend: PatchesOutputBackend):
        for i, j in tqdm(itertools.product(range(self.optical_rgb.shape[0] // self.patch_size[0]),
                        range(self.optical_rgb.shape[1] // self.patch_size[1]))):
            left_border = i * self.patch_size[0]
            right_border = (i + 1) * self.patch_size[0]
            top_border = j * self.patch_size[0]
            bottom_border = (j + 1) * self.patch_size[0]
            cur_patch = self.concatenate_full_patch(left_border, right_border, top_border, bottom_border)
            backend.save(array=cur_patch, label=0, path=self.dirs['all_patches'] + "/{}_{}.tif".format(i, j))

    def normalise(self):
        # to protect against evaluation of mean and vars for already normalised data
        # for example in notebooks where this code can be reused

        # todo compute mean for all normalisations
        self.original_optical_rgb = self.optical_rgb
        self.optical_rgb = self.optical_rgb.astype(np.float32)
        self.optical_rgb[:, :, 0] = self.optical_rgb[:, :, 0] / 255. - 0.5
        self.optical_rgb[:, :, 1] = self.optical_rgb[:, :, 1] / 255. - 0.5
        self.optical_rgb[:, :, 2] = self.optical_rgb[:, :, 2] / 255. - 0.5
        self.elevation_mean = np.mean(self.elevation)
        self.elevation_var = np.var(self.elevation)
        self.elevation = (self.elevation - self.elevation_mean) / self.elevation_var
        self.slope = (self.slope - 45) / 45

    def train_generator(self, batch_size):
        # todo add rotations etc
        while True:
            img_batch = np.zeros((batch_size,
                                  self.patch_size[0],
                                  self.patch_size[1],
                                  self.num_channels))
            lbl_batch = np.zeros((batch_size, 2))
            for i in range(batch_size):
                class_label = np.random.binomial(1, p=0.5, size=1)
                if class_label == 1:
                    img_batch[i] = self.sample_fault_patch()
                    lbl_batch[i] = np.array([0, 1])
                else:
                    img_batch[i] = self.sample_nonfault_patch()
                    lbl_batch[i] = np.array([1, 0])
            yield img_batch, lbl_batch
