import itertools
import logging
from enum import Enum
from typing import List, Tuple
import yaml
import io
import h5py
import numpy as np
from tqdm import trange

from src.DataPreprocessor.DataIOBackend.backend import DataIOBackend
from src.DataPreprocessor.PatchesOutputBackend.backend import PatchesOutputBackend
# from src.DataPreprocessor.image_augmentation import ImageAugmentation
from src.DataPreprocessor.normalised_data import NormalisedData
from src.config import data_preprocessor_params, areas, data_path

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
    BASIN_FAULT = 3
    STRIKE_SLIP_FAULT = 4
    THRUST_FAULT = 5


class RegionDataset:
    def __init__(self, region_id: int):
        self.region_id = region_id
        self.normalised_data = NormalisedData(region_id)
        self.trainable = None

        self.load()

    def load(self):
        logging.info('loading...')
        self.normalised_data.load()
        logging.info('loaded')

    def get_data_shape(self):
        return self.normalised_data.channels['optical_rgb'].shape[0], self.normalised_data.channels['optical_rgb'].shape[1], len(self.normalised_data.channels)

    def _borders_from_center(self, center, patch_size):
        left_border = center[0] - patch_size[0] // 2
        right_border = center[0] + patch_size[0] // 2
        top_border = center[1] - patch_size[1] // 2
        bottom_border = center[1] + patch_size[1] // 2

        im_width, im_height, _ = self.get_data_shape()

        if not (0 < left_border < im_width and 0 < right_border < im_width
                and 0 < top_border < im_height and 0 < bottom_border < im_height):
            raise OutOfBoundsException

        return left_border, right_border, top_border, bottom_border

    def concatenate_full_patch(self, left_border: int, right_border: int, top_border: int, bottom_border: int, channel_list: List[str]):
        np_channel_data = []
        for channel in channel_list:
            if self.normalised_data.channels[channel].ndim == 3:
                np_channel_data.append(self.normalised_data.channels[channel][
                                       left_border:right_border, top_border:bottom_border])
            else:
                np_channel_data.append(np.expand_dims(
                    self.normalised_data.channels[channel][
                        left_border:right_border, top_border:bottom_border],
                    axis=2))
        return np.concatenate(np_channel_data, axis=2)

    def get_full_image(self, channel_list):
        full_shape = self.get_data_shape()
        return self.concatenate_full_patch(left_border=0, right_border=full_shape[0], top_border=0, bottom_border=full_shape[1], channel_list=channel_list)

    def sample_fault_patch(self, patch_size):
        """if an image patch contains fault bit in the center area than assign it as a fault - go through fault lines
        and sample patches"""
        fault_locations = np.argwhere(self.normalised_data.features == FeatureValue.FAULT.value)
        sampled = False
        left_border, right_border, top_border, bottom_border = None, None, None, None
        while not sampled:
            samples_ind = np.random.randint(fault_locations.shape[0])
            try:
                left_border, right_border, top_border, bottom_border = self._borders_from_center(
                    fault_locations[samples_ind], patch_size)
                logging.info(
                    "extracting patch {}:{}, {}:{}".format(left_border, right_border, top_border, bottom_border))
                sampled = True
            except OutOfBoundsException:
                sampled = False

        coords = np.array((left_border, right_border, top_border, bottom_border))
        return self.concatenate_full_patch(left_border, right_border, top_border, bottom_border), coords

    def sample_fault_lookalike_patch(self, patch_size):
        """if an image patch contains fault lookalike bit in the center area than assign it as a fault - go through
        fault lookalike lines and sample patches"""
        fault_lookalike_locations = np.argwhere(self.normalised_data.features == FeatureValue.FAULT_LOOKALIKE.value)
        if fault_lookalike_locations.size == 0:
            logging.warning("no lookalikes marked, sampling nonfaults instead")
            return self.sample_nonfault_patch(patch_size)
        sampled = False
        left_border, right_border, top_border, bottom_border = None, None, None, None
        while not sampled:
            samples_ind = np.random.randint(fault_lookalike_locations.shape[0])
            try:
                left_border, right_border, top_border, bottom_border = self._borders_from_center(
                    fault_lookalike_locations[samples_ind], patch_size)
                logging.info(
                    "extracting patch {}:{}, {}:{}".format(left_border, right_border, top_border, bottom_border))
                sampled = True
            except OutOfBoundsException:
                sampled = False

        coords = np.array((left_border, right_border, top_border, bottom_border))
        return self.concatenate_full_patch(left_border, right_border, top_border, bottom_border), coords

    def sample_from_region(self, locations, inside_value, patch_size):
        # this can be changed for example to (50, 50) to reproduce experiment with
        # sampling non-fault patch as patch that may contain faults but not in the center
        # TODO make this a classmember or a method parameter
        inside_region_size = patch_size

        sampled = False
        left_border, right_border, top_border, bottom_border = None, None, None, None
        while not sampled:
            samples_ind = np.random.randint(locations.shape[0])
            try:
                left_border, right_border, top_border, bottom_border = self._borders_from_center(
                    locations[samples_ind], patch_size)
                logging.info(
                    "trying patch {}:{}, {}:{} as nonfault".format(left_border, right_border, top_border,
                                                                   bottom_border))
                is_probably_fault = False
                for i1, i2 in itertools.product(range(int(patch_size[0]/2 - inside_region_size[0]/2),
                                                      int(patch_size[0]/2 + inside_region_size[0]/2)),
                                                range(int(patch_size[1]/2 - inside_region_size[1]/2),
                                                      int(patch_size[1]/2 + inside_region_size[1]/2))):
                    if self.normalised_data.features[left_border + i1][top_border + i2] != inside_value:
                        is_probably_fault = True
                        logging.info("probably fault")
                        break
                if not is_probably_fault:
                    logging.info("nonfault")
                    sampled = True
            except OutOfBoundsException:
                sampled = False

        return left_border, right_border, top_border, bottom_border

    def sample_nonfault_patch(self, patch_size):
        """if an image path contains only nonfault bits, than assign it as a non-fault"""
        nonfault_locations = np.argwhere(self.normalised_data.features == FeatureValue.NONFAULT.value)
        if nonfault_locations.size == 0:
            logging.warning("no nonfaults marked, sampling undefined instead")
            return self.sample_undefined_patch(patch_size)

        left_border, right_border, top_border, bottom_border = \
            self.sample_from_region(nonfault_locations, FeatureValue.NONFAULT.value, patch_size)

        coords = np.array((left_border, right_border, top_border, bottom_border))
        return self.concatenate_full_patch(left_border, right_border, top_border, bottom_border), coords

    def sample_undefined_patch(self, patch_size):
        """if an image patch contains only undefined bits, than assign it as a undefined"""
        undefined_locations = np.argwhere(self.normalised_data.features == FeatureValue.UNDEFINED.value)
        if undefined_locations.size == 0:
            logging.warning("no undefined marked")
            raise Exception()

        left_border, right_border, top_border, bottom_border = \
            self.sample_from_region(undefined_locations, FeatureValue.UNDEFINED.value, patch_size)

        coords = np.array((left_border, right_border, top_border, bottom_border))
        return self.concatenate_full_patch(left_border, right_border, top_border, bottom_border), coords

    def sample_patch(self, label, patch_size):
        if label == FeatureValue.FAULT.value:
            return self.sample_fault_patch(patch_size)
        if label == FeatureValue.FAULT_LOOKALIKE.value:
            return self.sample_fault_lookalike_patch(patch_size)
        elif label == FeatureValue.NONFAULT.value:
            return self.sample_nonfault_patch(patch_size)
        else:
            raise NotImplementedError(f"class label {label}")

    # def sample_batch(self, batch_size, class_labels, patch_size, channels):
    #     # todo consider random preprocessing for rgb channels, such is tf.image.random_brightness, etc
    #     img_batch = np.zeros((batch_size,
    #                           patch_size[0],
    #                           patch_size[1],
    #                           channels.shape[0]))
    #
    #     coords_batch = np.zeros((batch_size, 4))
    #
    #     for i in range(batch_size):
    #         patch, coords = self.sample_patch(label=class_labels[i], patch_size=patch_size)
    #         augment_probability = np.random.rand()
    #         if augment_probability < 0.5:
    #             img_batch[i] = ImageAugmentation.augment(patch[:, :, channels])
    #         else:
    #             img_batch[i] = patch
    #         coords_batch[i] = coords
    #
    #     return img_batch, coords_batch

    def get_sample_3class(self, batch_size, class_probabilities, patch_size, channels, verbose:int):
        lbl_batch = np.zeros((batch_size, 3))
        class_labels = np.random.choice(class_probabilities.shape[0], batch_size, p=class_probabilities)

        img_batch, coords_batch = self.sample_batch(batch_size, class_labels, patch_size, channels)

        if verbose == 0:
            tqdm_disable = True
        elif verbose == 1:
            tqdm_disable = False
        else:
            raise ValueError()

        for i in trange(batch_size, disable=tqdm_disable):
            lbl_batch[i, class_labels[i]] = 1
        return img_batch, lbl_batch, coords_batch

    def get_sample_2class_lookalikes_with_nonfaults(self, batch_size, class_probabilities, patch_size, channels, verbose:int):
        lbl_batch = np.zeros((batch_size, 2))
        class_labels = np.random.choice(class_probabilities.shape[0], batch_size, p=class_probabilities)

        img_batch, coords_batch = self.sample_batch(batch_size, class_labels, patch_size, channels)

        if verbose == 0:
            tqdm_disable = True
        elif verbose == 1:
            tqdm_disable = False
        else:
            raise ValueError()

        for i in trange(batch_size, disable=tqdm_disable):
            if class_labels[i] == FeatureValue.NONFAULT.value or class_labels[i] == FeatureValue.FAULT_LOOKALIKE.value:
                lbl_batch[i, 1] = 1
            elif class_labels[i] == FeatureValue.FAULT.value:
                lbl_batch[i, 0] = 1
        return img_batch, lbl_batch, coords_batch

    def train_generator_3class(self, batch_size, class_probabilities, patch_size, channels, verbose:int):
        while True:
            img_batch, lbl_batch, coords_batch = self.get_sample_3class(batch_size, class_probabilities, patch_size, channels, verbose)
            yield img_batch.astype(np.float32), lbl_batch.astype(np.int32), coords_batch

    def train_generator_2class_lookalikes_with_nonfaults(self, batch_size, class_probabilities, patch_size, channels, verbose:int):
        #todo replace channels as array of ints with array of strings, as we have no explicit mapping of strings to ints
        while True:
            img_batch, lbl_batch, coords_batch = self.get_sample_2class_lookalikes_with_nonfaults(batch_size, class_probabilities,
                                                                                    patch_size, channels, verbose)
            yield img_batch, lbl_batch, coords_batch

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

