import itertools
from enum import Enum
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import logging
import pathlib
from tqdm import trange, tqdm

from src.DataPreprocessor.DataIOBackend.backend import DataIOBackend
from src.DataPreprocessor.PatchesOutputBackend.backend import PatchesOutputBackend


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


# todo add option to normalise based on normalisation features from a different data - or this problem is more serious,
# need to think, look at the distributions, probably we can't convert elevation this way for example

# todo consider creating another pipeline - that takes patches and outputs lines, not single probabilities. U-nets?
# todo add test/validation
class DataPreprocessor:
    def __init__(self, data_dir: str, data_io_backend: DataIOBackend, patches_output_backend: PatchesOutputBackend,
                 filename_prefix: str, mode: Mode, seed: int):
        np.random.seed(seed)
        self.data_dir = data_dir
        self.channels = dict(elevation=None, slope=None, optical_rgb=None, nir=None, ir=None, swir1=None, swir2=None,
                             panchromatic=None, curve=None, erosion=None)
        self.normalised_channels = dict(elevation=None, slope=None, optical_rgb=None, nir=None, ir=None, swir1=None,
                                        swir2=None, panchromatic=None, curve=None, erosion=None)
        self.features = None
        self.filename_prefix = filename_prefix
        self.mode = mode
        self.data_io_backend = data_io_backend
        self.patches_output_backend = patches_output_backend
        self.__load()
        self.__normalise()

    def __check_crop_data(self):
        min_shape = self.get_data_shape()[0], self.get_data_shape()[1]
        for channel in self.channels.values():
            min_shape = min(channel.shape[0], min_shape[0]), min(channel.shape[1], min_shape[1])

        for ch_name, channel in self.channels.items():
            if channel.shape[0] > min_shape[0] or channel.shape[1] > min_shape[1]:
                logging.warning("{} images do not match in size".format(ch_name))
                self.channels[ch_name] = channel[:min_shape[0], :min_shape[1]]

    def __check_crop_features(self):
        if self.features.shape[0] > self.get_data_shape()[0] or self.features.shape[1] > self.get_data_shape()[1]:
            logging.warning("features do not match in size")
            self.features = self.features[:self.get_data_shape()[0], :self.get_data_shape()[1]]

    def __load(self):
        logging.info('loading...')
        self.channels['elevation'] = self.data_io_backend.load_elevation(path=self.data_dir + self.filename_prefix + '_elev.tif')
        self.channels['slope'] = self.data_io_backend.load_slope(path=self.data_dir + self.filename_prefix + '_slope.tif')
        self.channels['optical_rgb'] = self.data_io_backend.load_optical(path_r=self.data_dir + self.filename_prefix + '_R.tif',
                                                             path_g=self.data_dir + self.filename_prefix + '_G.tif',
                                                             path_b=self.data_dir + self.filename_prefix + '_B.tif')
        self.channels['nir'] = self.data_io_backend.load_nir(path=self.data_dir + self.filename_prefix + '_NIR.tif')
        self.channels['ir'] = self.data_io_backend.load_nir(path=self.data_dir + self.filename_prefix + '_IR.tif')
        self.channels['swir1'] = self.data_io_backend.load_nir(path=self.data_dir + self.filename_prefix + '_SWIR1.tif')
        self.channels['swir2'] = self.data_io_backend.load_nir(path=self.data_dir + self.filename_prefix + '_SWIR2.tif')
        self.channels['panchromatic'] = self.data_io_backend.load_panchromatic(path=self.data_dir + self.filename_prefix + '_P.tif')
        try:
            self.channels['curve'] = self.data_io_backend.load_curve(path=self.data_dir + self.filename_prefix + '_curv.tif')
        except FileNotFoundError as err:
            logging.warning("Error: {}".format(err))
        try:
            self.channels['erosion'] = self.data_io_backend.load_erosion(path=self.data_dir + self.filename_prefix + '_erode.tif')
        except FileNotFoundError as err:
            logging.warning("Error: {}".format(err))
        self.__check_crop_data()

        if self.mode == Mode.TRAIN:
            self.features = self.data_io_backend.load_features(path=self.data_dir + 'feature_categories.tif')
            self.__check_crop_features()
        logging.info('loaded')

    def get_data_shape(self):
        return self.channels['optical_rgb'].shape[0], self.channels['optical_rgb'].shape[1], len(self.channels)

    def __borders_from_center(self, center, patch_size):
        left_border = center[0] - patch_size[0] // 2
        right_border = center[0] + patch_size[0] // 2
        top_border = center[1] - patch_size[1] // 2
        bottom_border = center[1] + patch_size[1] // 2

        im_width, im_height, _ = self.get_data_shape()

        if not (0 < left_border < im_width and 0 < right_border < im_width
                and 0 < top_border < im_height and 0 < bottom_border < im_height):
            raise OutOfBoundsException

        return left_border, right_border, top_border, bottom_border

    def concatenate_full_patch(self, left_border: int, right_border: int, top_border: int, bottom_border: int):
        return np.concatenate(
            (self.normalised_channels['optical_rgb'][left_border:right_border, top_border:bottom_border],
             np.expand_dims(self.normalised_channels['elevation'][left_border:right_border, top_border:bottom_border], axis=2),
             np.expand_dims(self.normalised_channels['slope'][left_border:right_border, top_border:bottom_border], axis=2),
             np.expand_dims(self.normalised_channels['nir'][left_border:right_border, top_border:bottom_border], axis=2),
             np.expand_dims(self.normalised_channels['ir'][left_border:right_border, top_border:bottom_border], axis=2),
             np.expand_dims(self.normalised_channels['swir1'][left_border:right_border, top_border:bottom_border], axis=2),
             np.expand_dims(self.normalised_channels['swir2'][left_border:right_border, top_border:bottom_border], axis=2),
             np.expand_dims(self.normalised_channels['panchromatic'][left_border:right_border, top_border:bottom_border], axis=2),
             np.expand_dims(self.normalised_channels['curve'][left_border:right_border, top_border:bottom_border], axis=2),
             np.expand_dims(self.normalised_channels['erosion'][left_border:right_border, top_border:bottom_border], axis=2)),
            axis=2)

    def sample_fault_patch(self, patch_size):
        """if an image patch contains fault bit in the center area than assign it as a fault - go through fault lines
        and sample patches"""
        fault_locations = np.argwhere(self.features == FeatureValue.FAULT.value)
        sampled = False
        left_border, right_border, top_border, bottom_border = None, None, None, None
        while not sampled:
            samples_ind = np.random.randint(fault_locations.shape[0])
            try:
                left_border, right_border, top_border, bottom_border = self.__borders_from_center(
                    fault_locations[samples_ind], patch_size)
                logging.info(
                    "extracting patch {}:{}, {}:{}".format(left_border, right_border, top_border, bottom_border))
                sampled = True
            except OutOfBoundsException:
                sampled = False

        return self.concatenate_full_patch(left_border, right_border, top_border, bottom_border)

    def sample_fault_lookalike_patch(self, patch_size):
        """if an image patch contains fault lookalike bit in the center area than assign it as a fault - go through
        fault lookalike lines and sample patches"""
        fault_locations = np.argwhere(self.features == FeatureValue.FAULT_LOOKALIKE.value)
        sampled = False
        left_border, right_border, top_border, bottom_border = None, None, None, None
        while not sampled:
            samples_ind = np.random.randint(fault_locations.shape[0])
            try:
                left_border, right_border, top_border, bottom_border = self.__borders_from_center(
                    fault_locations[samples_ind], patch_size)
                logging.info(
                    "extracting patch {}:{}, {}:{}".format(left_border, right_border, top_border, bottom_border))
                sampled = True
            except OutOfBoundsException:
                sampled = False

        return self.concatenate_full_patch(left_border, right_border, top_border, bottom_border)

    def sample_nonfault_patch(self, patch_size):
        """if an image path contains only nonfault bits, than assign it as a non-fault"""
        nonfault_locations = np.argwhere(self.features == FeatureValue.NONFAULT.value)
        sampled = False
        left_border, right_border, top_border, bottom_border = None, None, None, None
        while not sampled:
            samples_ind = np.random.randint(nonfault_locations.shape[0])
            try:
                left_border, right_border, top_border, bottom_border = self.__borders_from_center(
                    nonfault_locations[samples_ind], patch_size)
                logging.info(
                    "trying patch {}:{}, {}:{} as nonfault".format(left_border, right_border, top_border,
                                                                   bottom_border))
                is_probably_fault = False
                for i1, i2 in itertools.product(range(patch_size[0]), range(patch_size[1])):
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

    def sample_patch(self, label, patch_size):
        if label == FeatureValue.FAULT:
            return self.sample_fault_patch(patch_size)
        if label == FeatureValue.FAULT_LOOKALIKE:
            return self.sample_fault_lookalike_patch(patch_size)
        elif label == FeatureValue.NONFAULT:
            return self.sample_nonfault_patch(patch_size)

    def __normalise(self):
        # todo var -> std
        self.normalised_channels['optical_rgb'] = self.channels['optical_rgb'].astype(np.float32) / 255. - 0.5
        self.normalised_channels['elevation'] = (self.channels['elevation'].astype(np.float32) - np.mean(
            self.channels['elevation'].astype(np.float32))) / np.std(self.channels['elevation'].astype(np.float32))
        self.normalised_channels['slope'] = (self.channels['slope'].astype(np.float32) - 45.) / 45.
        self.normalised_channels['nir'] = (self.channels['nir'].astype(np.float32) - np.mean(
            self.channels['nir'].astype(np.float32))) / np.std(self.channels['nir'].astype(np.float32))
        self.normalised_channels['ir'] = (self.channels['ir'].astype(np.float32) - np.mean(
            self.channels['ir'].astype(np.float32))) / np.std(self.channels['ir'].astype(np.float32))
        self.normalised_channels['swir1'] = (self.channels['swir1'].astype(np.float32) - np.mean(
            self.channels['swir1'].astype(np.float32))) / np.std(self.channels['swir1'].astype(np.float32))
        self.normalised_channels['swir2'] = (self.channels['swir2'].astype(np.float32) - np.mean(
            self.channels['swir2'].astype(np.float32))) / np.std(self.channels['swir2'].astype(np.float32))
        self.normalised_channels['panchromatic'] = (self.channels['panchromatic'].astype(np.float32) - np.mean(
            self.channels['panchromatic'].astype(np.float32))) / np.std(self.channels['panchromatic'].astype(np.float32))
        self.normalised_channels['curve'] = (self.channels['curve'].astype(np.float32) - np.mean(
            self.channels['curve'].astype(np.float32))) / np.std(self.channels['curve'].astype(np.float32))
        self.normalised_channels['erosion'] = (self.channels['erosion'].astype(np.float32) - np.mean(
            self.channels['erosion'].astype(np.float32))) / np.std(self.channels['erosion'].astype(np.float32))

    def denormalise(self, patch):
        denormalised_rgb = ((patch[:, :, :3] + 0.5) * 255).astype(np.uint8)
        denormalised_elevation = (patch[:, :, 3] * np.std(self.channels['elevation'].astype(np.float32)) + np.mean(
            self.channels['elevation'].astype(np.float32)))
        denormalised_slope = (patch[:, :, 4] * 45 + 45)
        denormalised_nir = (patch[:, :, 5] * np.std(self.channels['nir'].astype(np.float32)) + np.mean(
            self.channels['nir'].astype(np.float32)))
        denormalised_ir = (patch[:, :, 6] * np.std(self.channels['nir'].astype(np.float32)) + np.mean(
            self.channels['nir'].astype(np.float32)))
        denormalised_swir1 = (patch[:, :, 7] * np.std(self.channels['nir'].astype(np.float32)) + np.mean(
            self.channels['nir'].astype(np.float32)))
        denormalised_swir2 = (patch[:, :, 8] * np.std(self.channels['nir'].astype(np.float32)) + np.mean(
            self.channels['nir'].astype(np.float32)))
        denormalised_panchromatic = (patch[:, :, 9] * np.std(self.channels['nir'].astype(np.float32)) + np.mean(
            self.channels['nir'].astype(np.float32)))
        return denormalised_rgb, denormalised_elevation, denormalised_slope

    def get_sample_3class(self, batch_size, class_probabilities, patch_size, channels):
        num_classes = class_probabilities.shape[0]
        img_batch = np.zeros((batch_size,
                              patch_size[0],
                              patch_size[1],
                              channels.shape[0]))
        lbl_batch = np.zeros((batch_size, num_classes))
        class_labels = np.random.choice(num_classes, batch_size, p=class_probabilities)

        for i in range(batch_size):
            if class_labels[i] == FeatureValue.FAULT.value:
                patch = self.sample_fault_patch(patch_size)
            elif class_labels[i] == FeatureValue.FAULT_LOOKALIKE.value:
                patch = self.sample_fault_lookalike_patch(patch_size)
            elif class_labels[i] == FeatureValue.NONFAULT.value:
                patch = self.sample_nonfault_patch(patch_size)
            else:
                raise NotImplementedError("class label {}".format(class_labels[i]))

            for _ in range(np.random.randint(0, 4)):
                patch = np.rot90(patch, axes=(0, 1))
            for _ in range(np.random.randint(0, 2)):
                patch = np.fliplr(patch)
            for _ in range(np.random.randint(0, 2)):
                patch = np.flipud(patch)

            img_batch[i] = patch[:, :, channels]
            lbl_batch[i, class_labels[i]] = 1
        return img_batch, lbl_batch

    def get_sample_2class_lookalikes_with_nonfaults(self, batch_size, class_probabilities, patch_size, channels):
        num_classes = class_probabilities.shape[0]
        img_batch = np.zeros((batch_size,
                              patch_size[0],
                              patch_size[1],
                              channels.shape[0]))
        lbl_batch = np.zeros((batch_size, 2))
        class_labels = np.random.choice(num_classes, batch_size, p=class_probabilities)

        for i in range(batch_size):
            if class_labels[i] == FeatureValue.FAULT.value:
                patch = self.sample_fault_patch(patch_size)
            elif class_labels[i] == FeatureValue.FAULT_LOOKALIKE.value:
                patch = self.sample_fault_lookalike_patch(patch_size)
            elif class_labels[i] == FeatureValue.NONFAULT.value:
                patch = self.sample_nonfault_patch(patch_size)
            else:
                raise NotImplementedError("class label {}".format(class_labels[i]))

            for _ in range(np.random.randint(0, 4)):
                patch = np.rot90(patch, axes=(0, 1))
            for _ in range(np.random.randint(0, 2)):
                patch = np.fliplr(patch)
            for _ in range(np.random.randint(0, 2)):
                patch = np.flipud(patch)

            img_batch[i] = patch[:, :, channels]
            if class_labels[i] == FeatureValue.NONFAULT.value or class_labels[i] == FeatureValue.FAULT_LOOKALIKE.value:
                lbl_batch[i, 1] = 1
            elif class_labels[i] == FeatureValue.FAULT.value:
                lbl_batch[i, 0] = 1
        return img_batch, lbl_batch

    def train_generator_3class(self, batch_size, class_probabilities, patch_size, channels):
        while True:
            img_batch, lbl_batch = self.get_sample_3class(batch_size, class_probabilities, patch_size, channels)
            yield img_batch.astype(np.float32), lbl_batch.astype(np.int32)

    def train_generator_2class_lookalikes_with_nonfaults(self, batch_size, class_probabilities, patch_size, channels):
        while True:
            img_batch, lbl_batch = self.get_sample_2class_lookalikes_with_nonfaults(batch_size, class_probabilities,
                                                                                    patch_size, channels)
            yield img_batch, lbl_batch

    def sequential_pass_generator(self, patch_size: Tuple[int, int], stride: int, batch_size: int, channels: List[int]):
        """note the different order of indexes in coords and patch ind, this was due to this input in tf non_max_suppression"""
        #todo consider returning views to reduce required memory. Check sklearn.feature_extraction.image.extract_patches
        batch_ind = 0
        patch_coords_batch = []
        patch_batch = []
        img_width, img_height, _ = self.get_data_shape()
        for top_left_border_x, top_left_border_y in itertools.product(
                range(0, img_width - patch_size[0], stride),
                range(0, img_height - patch_size[1], stride)):

            patch_coords_batch.append(np.array([top_left_border_x, top_left_border_y, top_left_border_x + patch_size[0],
                                                top_left_border_y + patch_size[1]]))
            patch_batch.append(
                self.concatenate_full_patch(top_left_border_x, top_left_border_x + patch_size[0], top_left_border_y,
                                            top_left_border_y + patch_size[1]))
            batch_ind = batch_ind + 1
            if batch_ind >= batch_size:
                patch_coords_batch_np = np.stack(patch_coords_batch, axis=0)
                patch_batch_np = np.stack(patch_batch, axis=0)
                yield patch_coords_batch_np, patch_batch_np[:, :, :, channels]
                batch_ind = 0
                patch_coords_batch = []
                patch_batch = []

        #yield last patch
        if len(patch_coords_batch) > 0:
            patch_coords_batch_np = np.stack(patch_coords_batch, axis=0)
            patch_batch_np = np.stack(patch_batch, axis=0)
            yield patch_coords_batch_np, patch_batch_np[:, :, :, channels]
