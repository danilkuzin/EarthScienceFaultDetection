import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
import os
import pathlib

from PIL import Image


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
        self.num_faults_train = 100
        self.num_nonfaults_train = 100

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
        self.optical_rgb = np.dstack((optical_r, optical_g, optical_b))
        # plt.imsave('data.tif', optical_rgb)

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

    def prepare_train(self):
        # if an image patch contains fault bit in the center area than assign it as a fault - go through fault lines and sample patches
        fault_locations = np.argwhere(self.features == 1)
        samples_ind = np.random.randint(fault_locations.shape[0], size=self.num_faults_train)
        for i in range(self.num_faults_train):
            # TODO check if out of original image borders and add this condition to argwhere?
            left_border = fault_locations[samples_ind[i]][0]-self.patch_size[0]//2
            right_border = fault_locations[samples_ind[i]][0]+self.patch_size[0]//2
            top_border = fault_locations[samples_ind[i]][1]-self.patch_size[1]//2
            bottom_border = fault_locations[samples_ind[i]][1]+self.patch_size[1]//2
            left_border, right_border, top_border, bottom_border = self.borders_from_center(fault_locations[samples_ind[i]])
            logging.info("extracting patch {}:{}, {}:{}".format(left_border, right_border, top_border, bottom_border))
            cur_patch = self.optical_rgb[left_border:right_border][top_border:bottom_border]
            plt.imsave(self.dirs['train_fault']+"/{}.tif".format(i), cur_patch)

        # if an image path contains only nonfault bits, than assign it as a non-fault
        for i in range(self.num_nonfaults_train):
            sampled = False
            while not sampled:
                ind1 = np.random.randint(self.patch_size[0], self.optical_rgb.shape[0] - self.patch_size[0])
                ind2 = np.random.randint(self.patch_size[0], self.optical_rgb.shape[0] - self.patch_size[0])
                left_border, right_border, top_border, bottom_border = self.borders_from_center(
                    fault_locations[samples_ind[i]])
                logging.info("trying patch {}:{}, {}:{} as nonfault".format(left_border, right_border, top_border, bottom_border))
                for i1 in range(self.patch_size[0]):
                    for i2 in range(self.patch_size[1]):
                        if self.features[left_border + i1][top_border + i2] != 3:
                            continue
                cur_patch = self.optical_rgb[left_border:right_border][top_border:bottom_border]
                plt.imsave(self.dirs['train_nonfault'] + "/{}.tif".format(i), cur_patch)
                sampled = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    loader = DataPreprocessor22012019("../../data/Data22012019/")
    loader.prepare_folders()
    loader.load()
    loader.prepare_train()

