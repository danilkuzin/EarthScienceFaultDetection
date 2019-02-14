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


#todo maybe add script to convert filenames from drive to standard names without prefixes
#todo add option to tormalise based on normalisation features from a different data - or this problem is more serious,
# need to think, look at the distributions, probably we can't convert elevation this way for example
#todo move output backend into init, add in-memory backend, move the directories outputs to other backends than in-memory
#todo finish adding infrared images
class DataPreprocessor:
    def __init__(self, data_dir: str, data_io_backend: DataIOBackend, patches_output_backend: PatchesOutputBackend,
                 filename_prefix: str, mode: Mode, seed: int):
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
        self.num_channels = 5 # r, g, b, elevation, slope
        self.filename_prefix = filename_prefix
        self.mode = mode
        self.data_io_backend = data_io_backend
        self.patches_output_backend = patches_output_backend
        self.load()
        self.normalised_elevation = None
        self.normalised_slope = None
        self.normalised_optical_rgb = None
        self.normalise()

    def check_crop_data(self):
        #todo consider moving each band into dict of bands so that we can iterate over them here
        min_size = 0
        for i in range(2):
            min_size = min(self.optical_rgb.shape[i], self.elevation.shape[i], self.slope.shape[i])

        if min(self.optical_rgb.shape[0], self.optical_rgb.shape[1]) > min_size:
            logging.warning("optical images are not match in size")
            self.optical_rgb = self.optical_rgb[:min_size, :min_size]
        if min(self.elevation.shape[0], self.elevation.shape[1]) > min_size:
            logging.warning("elevation images are not match in size")
            self.elevation = self.elevation[:min_size, :min_size]
        if min(self.slope.shape[0], self.slope.shape[1]) > min_size:
            logging.warning("slope images are not match in size")
            self.slope = self.slope[:min_size, :min_size]

    def load(self):
        logging.info('loading...')
        self.elevation = self.data_io_backend.load_elevation(path=self.data_dir + self.filename_prefix + '_elev.tif')
        self.slope = self.data_io_backend.load_slope(path=self.data_dir + self.filename_prefix + '_slope.tif')
        self.optical_rgb = self.data_io_backend.load_optical(path_r=self.data_dir + self.filename_prefix + '_R.tif',
                                                path_g=self.data_dir + self.filename_prefix + '_G.tif',
                                                path_b=self.data_dir + self.filename_prefix + '_B.tif')
        self.check_crop_data()
        plt.imsave(self.data_dir+'data.tif', self.optical_rgb)
        if self.mode == Mode.TRAIN:
            self.features = self.data_io_backend.load_features(path=self.data_dir+'feature_categories.tif')
        logging.info('loaded')

    def borders_from_center(self, center, patch_size):
        left_border = center[0] - patch_size[0] // 2
        right_border = center[0] + patch_size[0] // 2
        top_border = center[1] - patch_size[1] // 2
        bottom_border = center[1] + patch_size[1] // 2

        if not (0 < left_border < self.optical_rgb.shape[0]
                and 0 < right_border < self.optical_rgb.shape[0]
                and 0 < top_border < self.optical_rgb.shape[1]
                and 0 < bottom_border < self.optical_rgb.shape[1]):
            raise OutOfBoundsException

        return left_border, right_border, top_border, bottom_border

    def concatenate_full_patch(self, left_border: int, right_border: int, top_border: int, bottom_border: int):
        return np.concatenate((self.normalised_optical_rgb[left_border:right_border, top_border:bottom_border],
                               np.expand_dims(self.normalised_elevation[left_border:right_border, top_border:bottom_border], axis=2),
                               np.expand_dims(self.normalised_slope[left_border:right_border, top_border:bottom_border], axis=2)),
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
                left_border, right_border, top_border, bottom_border = self.borders_from_center(
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
                left_border, right_border, top_border, bottom_border = self.borders_from_center(
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
                left_border, right_border, top_border, bottom_border = self.borders_from_center(
                    nonfault_locations[samples_ind], patch_size)
                logging.info(
                    "trying patch {}:{}, {}:{} as nonfault".format(left_border, right_border, top_border, bottom_border))
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
        if label==FeatureValue.FAULT:
            return self.sample_fault_patch(patch_size)
        if label==FeatureValue.FAULT_LOOKALIKE:
            return self.sample_fault_lookalike_patch(patch_size)
        elif label==FeatureValue.NONFAULT:
            return self.sample_nonfault_patch(patch_size)

    def normalise(self):
        self.normalised_optical_rgb = self.optical_rgb.astype(np.float32)
        self.normalised_optical_rgb[:, :, 0] = self.optical_rgb[:, :, 0] / 255. - 0.5
        self.normalised_optical_rgb[:, :, 1] = self.optical_rgb[:, :, 1] / 255. - 0.5
        self.normalised_optical_rgb[:, :, 2] = self.optical_rgb[:, :, 2] / 255. - 0.5
        self.elevation_mean = np.mean(self.elevation)
        self.elevation_var = np.var(self.elevation)
        self.normalised_elevation = (self.elevation - self.elevation_mean) / self.elevation_var
        self.normalised_slope = (self.slope - 45) / 45

    def denormalise(self, patch):
        denormalised_rgb = ((patch[:,:,:3]+0.5) * 255).astype(np.uint8)
        denormalised_elevation = (patch[:,:,3]*self.elevation_var + self.elevation_mean)
        denormalised_slope = (patch[:,:,4]*45 + 45)
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
            elif  class_labels[i] == FeatureValue.FAULT.value:
                lbl_batch[i, 0] = 1
        return img_batch, lbl_batch

    def train_generator_3class(self, batch_size, class_probabilities, patch_size, channels):
        while True:
            img_batch, lbl_batch = self.get_sample_3class(batch_size, class_probabilities, patch_size, channels)
            yield img_batch, lbl_batch

    def train_generator_2class_lookalikes_with_nonfaults(self, batch_size, class_probabilities, patch_size, channels):
        while True:
            img_batch, lbl_batch = self.get_sample_2class_lookalikes_with_nonfaults(batch_size, class_probabilities, patch_size, channels)
            yield img_batch, lbl_batch

    def sequential_pass_generator(self, patch_size: Tuple[int, int], stride: int, batch_size:int, channels:List[int]):
        """note the different order of indexes in coords and patch ind, this was due to this input in tf non_max_suppression"""
        batch_ind = 0
        patch_coords_batch = []
        patch_batch = []
        for top_left_border_x, top_left_border_y in itertools.product(range(0, self.optical_rgb.shape[0] - patch_size[0], stride),
                                                                          range(0, self.optical_rgb.shape[1] - patch_size[1], stride)):

            patch_coords_batch.append(np.array([top_left_border_x, top_left_border_y, top_left_border_x + patch_size[0],
                                       top_left_border_y + patch_size[1]]))
            patch_batch.append(self.concatenate_full_patch(top_left_border_x, top_left_border_x + patch_size[0], top_left_border_y, top_left_border_y + patch_size[1]))
            batch_ind = batch_ind + 1
            if batch_ind >= batch_size:
                patch_coords_batch_np = np.stack(patch_coords_batch, axis=0)
                patch_batch_np = np.stack(patch_batch, axis=0)
                yield patch_coords_batch_np, patch_batch_np[:, :, :, channels]
                batch_ind = 0
                patch_coords_batch = []
                patch_batch = []


