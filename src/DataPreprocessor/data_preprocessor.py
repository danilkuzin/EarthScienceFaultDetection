import itertools
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import logging
import pathlib
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
    UNDEFINED = -1
    FAULT = 0
    FAULT_LOOKALIKE = 1
    NONFAULT = 2

class DatasetType(Enum):
    TRAIN = 1,
    VALIDATION = 2,
    TEST = 3

#todo add option to tormalise based on normalisation features from a different data
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
        self.prepare_folders()
        self.load(backend)
        self.normalised_elevation = None
        self.normalised_slope = None
        self.normalised_optical_rgb = None
        self.normalise()

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

    def prepare_datasets(self, output_backend, patch_size):
        self.prepare_dataset(output_backend, DatasetType.TRAIN, FeatureValue.FAULT, patch_size)
        self.prepare_dataset(output_backend, DatasetType.TRAIN, FeatureValue.NONFAULT, patch_size)
        self.prepare_dataset(output_backend, DatasetType.TRAIN, FeatureValue.FAULT_LOOKALIKE, patch_size)
        self.prepare_dataset(output_backend, DatasetType.VALIDATION, FeatureValue.FAULT, patch_size)
        self.prepare_dataset(output_backend, DatasetType.VALIDATION, FeatureValue.NONFAULT, patch_size)
        self.prepare_dataset(output_backend, DatasetType.VALIDATION, FeatureValue.FAULT_LOOKALIKE, patch_size)
        self.prepare_dataset(output_backend, DatasetType.TEST, FeatureValue.FAULT, patch_size)
        self.prepare_dataset(output_backend, DatasetType.TEST, FeatureValue.NONFAULT, patch_size)
        self.prepare_dataset(output_backend, DatasetType.TEST, FeatureValue.FAULT_LOOKALIKE, patch_size)

    def prepare_dataset(self, output_backend: PatchesOutputBackend, data_type, label, patch_size):
        category = data_type.name + "_" + label.name
        arr = np.zeros(
            (self.datasets_sizes[category], patch_size[0], patch_size[1], self.num_channels))
        for i in trange(self.datasets_sizes[category]):
            arr[i] = self.sample_patch(label)
        output_backend.save(arr, label==1 if 0 else 1, self.dirs[category])

    def prepare_all_patches(self, backend: PatchesOutputBackend, patch_size):
        for i, j in tqdm(itertools.product(range(self.optical_rgb.shape[0] // patch_size[0]),
                        range(self.optical_rgb.shape[1] // patch_size[1]))):
            left_border = i * patch_size[0]
            right_border = (i + 1) * patch_size[0]
            top_border = j * patch_size[0]
            bottom_border = (j + 1) * patch_size[0]
            cur_patch = self.concatenate_full_patch(left_border, right_border, top_border, bottom_border)
            backend.save(array=cur_patch, label=0, path=self.dirs['all_patches'] + "/{}_{}.tif".format(i, j))

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

    def train_generator(self, batch_size, class_probabilities, patch_size, channels):
        num_classes = class_probabilities.shape[0]
        while True:
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
            yield img_batch, lbl_batch
