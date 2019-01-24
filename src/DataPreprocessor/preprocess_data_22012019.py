import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
import os
import pathlib

from PIL import Image

#TODO rewrite this as some tf.Dataset.from_generator or keras.ImageDataGenerator that feeds data in the same manner
from tqdm import trange


class DataPreprocessor22012019:
    def __init__(self, data_dir):
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
        self.patch_size = (150, 150)
        self.center_size = (50, 50)
        self.dirs = dict()
        self.num_faults_train = 10000
        self.num_nonfaults_train = 10000
        self.num_faults_valid = 1000
        self.num_nonfaults_valid = 1000
        self.num_test = 10
        self.true_test_classes = None
        self.load()

    def prepare_folders(self):
        self.dirs['train_fault'] = self.data_dir + "learn/train/fault/"
        self.dirs['train_nonfault'] = self.data_dir + "learn/train/nonfault/"
        self.dirs['valid_fault'] = self.data_dir + "learn/valid/fault/"
        self.dirs['valid_nonfault'] = self.data_dir + "learn/valid/nonfault/"
        self.dirs['test'] = self.data_dir + "learn/test/test/"
        pathlib.Path(self.dirs['train_fault']).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.dirs['train_nonfault']).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.dirs['valid_fault']).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.dirs['valid_nonfault']).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.dirs['test']).mkdir(parents=True, exist_ok=True)

    def load(self):
        logging.info('loading...')
        self.elevation = cv2.imread(self.data_dir+'tibet_elev.tif') # elevation map (from the Shuttle Radar Topography Mission), values in meters above sea level
        self.slope = cv2.imread(self.data_dir+'tibet_slope.tif' ) # slope map derived from the elevation, values in degrees from horizontal, 0-90
        optical_r = cv2.imread(self.data_dir+'tibet_R.tif') # standard red / green / blue optical bands from the Landsat-8 platform, each 0 - 255
        optical_g = cv2.imread(self.data_dir+'tibet_G.tif')
        optical_b = cv2.imread(self.data_dir+'tibet_B.tif')
        self.optical_rgb = np.dstack((optical_r[:,:,0], optical_g[:,:,0], optical_b[:,:,0]))
        #plt.imsave(self.data_dir+'data.tif', self.optical_rgb)

        self.nir = cv2.imread(self.data_dir+'tibet_NIR.tif') # near infrared from Landsat
        self.ir = cv2.imread(self.data_dir+'tibet_IR.tif') # infrared from Landsat
        self.swir1 = cv2.imread(self.data_dir+'tibet_SWIR1.tif') # shortwave infrared1 from Landsat
        self.swir2 = cv2.imread(self.data_dir+'tibet_SWIR2.tif') # shortwave infrared2 from Landsat
        self.panchromatic = cv2.imread(self.data_dir+'tibet_P.tif') # panchromatic band from Landsat, essentially just total surface reflectance, like a grayscale image of the ground
        features_map = Image.open(self.data_dir+'feature_categories.tif')
        self.features = np.array(features_map) #0 - neutral, undefined content (could include faults--fair area for testing)
        # 1 - faults
        # 2 - fault lookalikes - features that we think share visual or topographic similarity with faults, but expert interpretation can exclude
        # 3 - not-faults - areas that definitely do not include faults, nor things that we think even look like faults, can be used directly for training what faults are not.
        # features[features > 0] = 1
        # plt.imsave('features.png', features*255)
        logging.info('loaded')

    def borders_from_center(self, center):
        left_border = center[0] - self.patch_size[0] // 2
        right_border = center[0] + self.patch_size[0] // 2
        top_border = center[1] - self.patch_size[1] // 2
        bottom_border = center[1] + self.patch_size[1] // 2
        return left_border, right_border, top_border, bottom_border

    def sample_fault_patch(self):
        # if an image patch contains fault bit in the center area than assign it as a fault - go through fault lines and sample patches
        fault_locations = np.argwhere(self.features == 1)
        sampled = False
        while not sampled:
            samples_ind = np.random.randint(fault_locations.shape[0])
            left_border, right_border, top_border, bottom_border = self.borders_from_center(
                fault_locations[samples_ind])
            if 0 < left_border < self.optical_rgb.shape[0] \
                    and 0 < right_border < self.optical_rgb.shape[0] \
                    and 0 < top_border < self.optical_rgb.shape[1] \
                    and 0 < bottom_border < self.optical_rgb.shape[1]:
                sampled = True

            if sampled:
                logging.info(
                    "extracting patch {}:{}, {}:{}".format(left_border, right_border, top_border, bottom_border))
                return self.optical_rgb[left_border:right_border, top_border:bottom_border]

    def sample_nonfault_patch(self):
        # if an image path contains only nonfault bits, than assign it as a non-fault
        sampled = False
        while not sampled:
            # todo sample first index from non-faults
            ind1 = np.random.randint(self.patch_size[0], self.optical_rgb.shape[0] - self.patch_size[0])
            ind2 = np.random.randint(self.patch_size[0], self.optical_rgb.shape[0] - self.patch_size[0])
            left_border, right_border, top_border, bottom_border = self.borders_from_center(
                (ind1, ind2))
            logging.info(
                "trying patch {}:{}, {}:{} as nonfault".format(left_border, right_border, top_border, bottom_border))
            isProbablyFault = False
            for i1, i2 in zip(range(self.patch_size[0]), range(self.patch_size[1])):
                if self.features[left_border + i1][top_border + i2] != 3:
                    isProbablyFault = True
                    logging.info("probably fault")
                    break
            if not isProbablyFault:
                logging.info("nonfault")
                sampled = True
                return self.optical_rgb[left_border:right_border, top_border:bottom_border]

    def prepare_train(self):
        for i in trange(self.num_faults_train):
            cur_patch = self.sample_fault_patch()
            plt.imsave(self.dirs['train_fault'] + "/{}.tif".format(i), cur_patch)

        for i in trange(self.num_nonfaults_train):
            cur_patch = self.sample_nonfault_patch()
            plt.imsave(self.dirs['train_nonfault'] + "/{}.tif".format(i), cur_patch)

    def prepare_valid(self):
        for i in trange(self.num_faults_valid):
            cur_patch = self.sample_fault_patch()
            plt.imsave(self.dirs['valid_fault'] + "/{}.tif".format(i), cur_patch)

        for i in trange(self.num_nonfaults_valid):
            cur_patch = self.sample_nonfault_patch()
            plt.imsave(self.dirs['valid_nonfault'] + "/{}.tif".format(i), cur_patch)

    def prepare_test(self):
        self.true_test_classes = np.random.binomial(1, p=0.5, size=self.num_test)
        for i in trange(self.num_test):
            if self.true_test_classes[i] == 1:
                cur_patch = self.sample_fault_patch()
            else:
                cur_patch = self.sample_nonfault_patch()
            plt.imsave(self.dirs['test'] + "/{}.tif".format(i), cur_patch)
        logging.info("true test classes: {}".format(self.true_test_classes))

    def __iter__(self):
        while True:
            class_label = np.random.binomial(1, p=0.5, size=1)
            if class_label == 1:
                cur_patch = self.sample_fault_patch()
            else:
                cur_patch = self.sample_nonfault_patch()
            yield cur_patch, class_label


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    loader = DataPreprocessor22012019("../../data/Data22012019/")
    loader.prepare_folders()
    #loader.load()
    loader.prepare_train()
    loader.prepare_valid()
    loader.prepare_test()


