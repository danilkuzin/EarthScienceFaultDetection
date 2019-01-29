import h5py
import itertools
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
import os
import pathlib
import gdal
from osgeo import gdal_array
import struct

from PIL import Image

#TODO rewrite this as some tf.Dataset.from_generator or keras.ImageDataGenerator that feeds data in the same manner
from tqdm import trange, tqdm


class OutOfBoundsException(Exception):
    pass


class GdalFileException(Exception):
    pass

class DataOutput(Enum):
    FOLDER = 1,
    H5=2

class Backend(Enum):
    PILLOW = 1
    OPENCV = 2
    GDAL = 3

class FeatureValue(Enum):
    UNDEFINED = 0
    FAULT = 1
    FAULT_LOOKALIKE = 2
    NONFAULT = 3

class DatasetType(Enum):
    TRAIN = 1,
    VALIDATION = 2,
    TEST = 3

# todo normalise images
# todo include lookalikes as well
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
        self.datasets_sizes = dict()
        self.datasets_sizes[DatasetType.TRAIN.name + "_" + FeatureValue.FAULT.name] = 1000
        self.datasets_sizes[DatasetType.TRAIN.name + "_" + FeatureValue.NONFAULT.name] = 1000
        self.datasets_sizes[DatasetType.VALIDATION.name + "_" + FeatureValue.FAULT.name] = 200
        self.datasets_sizes[DatasetType.VALIDATION.name + "_" + FeatureValue.NONFAULT.name] = 200
        self.datasets_sizes[DatasetType.TEST.name + "_" + FeatureValue.FAULT.name] = 100
        self.datasets_sizes[DatasetType.TEST.name + "_" + FeatureValue.NONFAULT.name] = 100
        self.num_channels = 5 # r, g, b, elevation, slope
        self.true_test_classes = None
        self.load()

    def prepare_folders(self):
        self.dirs['train_fault'] = self.data_dir + "learn/train/fault/"
        self.dirs['train_nonfault'] = self.data_dir + "learn/train/nonfault/"
        self.dirs['valid_fault'] = self.data_dir + "learn/valid/fault/"
        self.dirs['valid_nonfault'] = self.data_dir + "learn/valid/nonfault/"
        self.dirs['test_w_labels_fault'] = self.data_dir + "learn/test_with_labels/fault/"
        self.dirs['test_w_labels_nonfault'] = self.data_dir + "learn/test_with_labels/nonfault/"
        self.dirs['test'] = self.data_dir + "learn/test/test/"
        self.dirs['all_patches'] = self.data_dir + "all/"
        pathlib.Path(self.dirs['train_fault']).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.dirs['train_nonfault']).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.dirs['valid_fault']).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.dirs['valid_nonfault']).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.dirs['test_w_labels_fault']).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.dirs['test_w_labels_nonfault']).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.dirs['test']).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.dirs['all_patches']).mkdir(parents=True, exist_ok=True)

    def load_elevation(self, backend):
        """elevation map (from the Shuttle Radar Topography Mission), values in meters above sea level"""
        path = self.data_dir + 'tibet_elev.tif'
        if backend == Backend.PILLOW:
            self.elevation = np.array(Image.open(path))
        elif backend == Backend.OPENCV:
            self.elevation = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        elif backend == Backend.GDAL:
            self.elevation = np.array(gdal.Open(path, gdal.GA_ReadOnly).ReadAsArray())

    def load_slope(self, backend):
        """slope map derived from the elevation, values in degrees from horizontal, 0-90"""
        path = self.data_dir + 'tibet_slope.tif'
        if backend == Backend.PILLOW:
            self.slope = np.array(Image.open(path))
        elif backend == Backend.OPENCV:
            #todo check why it produces None image
            self.slope = cv2.imread(path, cv2.IMREAD_LOAD_GDAL)
            raise NotImplementedError("currently not supported")
        elif backend == Backend.GDAL:
            self.slope = np.array(gdal.Open(path, gdal.GA_ReadOnly).ReadAsArray())

    def load_optical(self, backend):
        """standard red / green / blue optical bands from the Landsat-8 platform, each 0 - 255"""
        path_r, path_g, path_b = self.data_dir + 'tibet_R.tif', self.data_dir + 'tibet_G.tif', self.data_dir + 'tibet_B.tif'
        optical_r, optical_g, optical_b = None, None, None
        if backend == Backend.PILLOW:
            # todo check this
            raise NotImplementedError("currently not supported")
        elif backend == Backend.OPENCV:
            optical_r = cv2.imread(path_r)[:, :, 0]
            optical_g = cv2.imread(path_g)[:, :, 0]
            optical_b = cv2.imread(path_b)[:, :, 0]
        elif backend == Backend.GDAL:
            opt_string = '-ot Byte -of GTiff -scale 0 65535 0 255'
            # todo check how to remove tmp file
            dataset_r = gdal.Translate(self.data_dir + 'tmp.tif', gdal.Open(path_r, gdal.GA_ReadOnly),
                                       options=opt_string)
            optical_r = np.array(dataset_r.ReadAsArray())
            dataset_g = gdal.Translate(self.data_dir + 'tmp.tif', gdal.Open(path_g, gdal.GA_ReadOnly),
                                       options=opt_string)
            optical_g = np.array(dataset_g.ReadAsArray())
            dataset_b = gdal.Translate(self.data_dir + 'tmp.tif', gdal.Open(path_b, gdal.GA_ReadOnly),
                                       options=opt_string)
            optical_b = np.array(dataset_b.ReadAsArray())

        self.optical_rgb = np.dstack((optical_r, optical_g, optical_b))
        logging.warning("optical images are not match in 1-2 pixels in size")
        self.optical_rgb = self.optical_rgb[:self.elevation.shape[0], :self.elevation.shape[1]]

    def load_ir(self, backend):
        #todo add support for other backends
        self.nir = cv2.imread(self.data_dir + 'tibet_NIR.tif')  # near infrared from Landsat
        self.ir = cv2.imread(self.data_dir + 'tibet_IR.tif')  # infrared from Landsat
        self.swir1 = cv2.imread(self.data_dir + 'tibet_SWIR1.tif')  # shortwave infrared1 from Landsat
        self.swir2 = cv2.imread(self.data_dir + 'tibet_SWIR2.tif')  # shortwave infrared2 from Landsat
        self.panchromatic = cv2.imread(
        self.data_dir + 'tibet_P.tif')  # panchromatic band from Landsat, essentially just total surface reflectance, like a grayscale image of the ground

    def load_features(self, backend):
        """0 - neutral, undefined content (could include faults--fair area for testing)
        1 - faults
        2 - fault lookalikes - features that we think share visual or topographic similarity with faults, but expert interpretation can exclude
        3 - not-faults - areas that definitely do not include faults, nor things that we think even look like faults, can be used directly for training what faults are not."""
        path = self.data_dir+'feature_categories.tif'
        if backend == Backend.PILLOW:
            self.features = np.array(Image.open(path))
        elif backend == Backend.OPENCV:
            self.features = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        elif backend == Backend.GDAL:
            self.features = np.array(gdal.Open(path, gdal.GA_ReadOnly).ReadAsArray())

    def load(self, backend=Backend.GDAL):
        logging.info('loading...')
        self.load_elevation(backend)
        self.load_slope(backend)
        self.load_optical(backend)
        plt.imsave(self.data_dir+'data.tif', self.optical_rgb)
        self.load_ir(backend)
        self.load_features(backend)
        logging.info('loaded')

    # to be used for parsing gdal headers and recreating them in output results
    def parse_meta_with_gdal(self):
        # based on https://www.gdal.org/gdal_tutorial.html
        #Opening the File
        dataset = gdal.Open(self.data_dir+'tibet_R.tif', gdal.GA_ReadOnly)
        if not dataset:
            raise GdalFileException()

        scale = '-scale 0 65535 0 255'
        options_list = [
            '-ot Byte',
            '-of GTiff',
            scale
        ]
        options_string = " ".join(options_list)

        dataset = gdal.Translate(self.data_dir+'tmp.tif', dataset, options=options_string)

        #arr = dataset.ReadAsArray()

        # Getting Dataset Information
        print("Driver: {}/{}".format(dataset.GetDriver().ShortName,
                                     dataset.GetDriver().LongName))
        print("Size is {} x {} x {}".format(dataset.RasterXSize,
                                            dataset.RasterYSize,
                                            dataset.RasterCount))
        print("Projection is {}".format(dataset.GetProjection()))
        geotransform = dataset.GetGeoTransform()
        if geotransform:
            print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
            print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))

        # Fetching a Raster Band
        band = dataset.GetRasterBand(1)
        print("Band Type={}".format(gdal.GetDataTypeName(band.DataType)))

        min = band.GetMinimum()
        max = band.GetMaximum()
        if not min or not max:
            (min, max) = band.ComputeRasterMinMax(True)
        print("Min={:.3f}, Max={:.3f}".format(min, max))

        if band.GetOverviewCount() > 0:
            print("Band has {} overviews".format(band.GetOverviewCount()))

        if band.GetRasterColorTable():
            print("Band has a color table with {} entries".format(band.GetRasterColorTable().GetCount()))

        #dataset = gdal.Translate(self.data_dir+'tmp.tif', dataset, options=gdal.TranslateOptions(outputType=gdal.GDT_Byte, scaleParams=[0, 65535, 0, 255]))
        #dataset = gdal.Translate(self.data_dir+'tmp.tif', gdal.TranslateOptions(["-of", "GTiff", "-ot", "Byte", "-scale", "0 65535 0 255"]))

        # Reading Raster Data
        scanline = band.ReadRaster(xoff=0, yoff=0,
                                   xsize=band.XSize, ysize=1,
                                   buf_xsize=band.XSize, buf_ysize=1,
                                   buf_type=gdal.GDT_Byte)

        #tuple_of_floats = struct.unpack('f' * band.XSize, scanline)
        tuple_of_floats = struct.unpack('b' * band.XSize, scanline)

        dataset = None

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

    def sample_fault_patch(self):
        """if an image patch contains fault bit in the center area than assign it as a fault - go through fault lines
        and sample patches"""
        fault_locations = np.argwhere(self.features == 1)
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

        return np.concatenate((self.optical_rgb[left_border:right_border, top_border:bottom_border],
                               np.expand_dims(self.elevation[left_border:right_border, top_border:bottom_border], axis=2),
                               np.expand_dims(self.slope[left_border:right_border, top_border:bottom_border], axis=2)),
                              axis=2)

    def sample_nonfault_patch(self):
        """if an image path contains only nonfault bits, than assign it as a non-fault"""
        nonfault_locations = np.argwhere(self.features == 3)
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
                    if self.features[left_border + i1][top_border + i2] != 3:
                        is_probably_fault = True
                        logging.info("probably fault")
                        break
                if not is_probably_fault:
                    logging.info("nonfault")
                    sampled = True
            except OutOfBoundsException:
                sampled = False
        return np.concatenate((self.optical_rgb[left_border:right_border, top_border:bottom_border],
                               np.expand_dims(self.elevation[left_border:right_border, top_border:bottom_border], axis=2),
                               np.expand_dims(self.slope[left_border:right_border, top_border:bottom_border], axis=2)),
                              axis=2)

    def sample_patch(self, label):
        if label==FeatureValue.FAULT:
            return self.sample_fault_patch()
        elif label==FeatureValue.NONFAULT:
            return self.sample_nonfault_patch()

    def prepare_datasets(self, output):
        self.prepare_dataset(output, DatasetType.TRAIN, FeatureValue.FAULT)
        self.prepare_dataset(output, DatasetType.TRAIN, FeatureValue.NONFAULT)
        self.prepare_dataset(output, DatasetType.VALIDATION, FeatureValue.FAULT)
        self.prepare_dataset(output, DatasetType.VALIDATION, FeatureValue.NONFAULT)
        self.prepare_dataset(output, DatasetType.TEST, FeatureValue.FAULT)
        self.prepare_dataset(output, DatasetType.TEST, FeatureValue.NONFAULT)

    def prepare_dataset(self, output, data_type, label):
        category = data_type.name + "_" + label.name
        if output == DataOutput.FOLDER:
            for i in trange(self.datasets_sizes[category]):
                patch_im = Image.fromarray(self.sample_patch(label))
                patch_im.save(self.dirs[category] + "/{}.tif".format(i))

        elif output == DataOutput.H5:
            arr = np.zeros(
                (self.datasets_sizes[category], self.patch_size[0], self.patch_size[1], self.num_channels))
            for i in trange(self.datasets_sizes[category]):
                arr[i] = self.sample_patch(label)
            with h5py.File(self.data_dir + category + '.h5', 'w') as hf:
                hf.create_dataset(category, data=arr)

    def prepare_all_patches(self):
        for i, j in tqdm(itertools.product(range(self.optical_rgb.shape[0] // self.patch_size[0]),
                        range(self.optical_rgb.shape[1] // self.patch_size[1]))):
            cur_patch = self.optical_rgb[i * self.patch_size[0]: (i + 1) * self.patch_size[0],
                        j * self.patch_size[0]: (j + 1) * self.patch_size[0]]
            plt.imsave(self.dirs['all_patches'] + "/{}_{}.tif".format(i, j), cur_patch)

    #to be removed
    def combine_features_images(self, ):
        mask = Image.open(self.data_dir+'feature_categories.tif').convert('RGBA').crop((0, 0, 22 * 150, 22 * 150))
        mask_np = np.array(mask)
        for i1, i2 in tqdm(itertools.product(range(22 * 150), range(22 * 150))):
            if np.any(mask_np[i1, i2] == 1):
                mask_np[i1, i2] = [250, 0, 0, 0]
            if np.any(mask_np[i1, i2] == 2):
                mask_np[i1, i2] = [0, 250, 0, 0]
            if np.any(mask_np[i1, i2] == 3):
                mask_np[i1, i2] = [0, 0, 250, 0]
        mask_np[:, :, 3] = 60 * np.ones((22 * 150, 22 * 150))
        mask_a = Image.fromarray(mask_np)
        orig = Image.open(self.data_dir + 'data.tif')
        orig_c = orig.crop((0, 0, 22 * 150, 22 * 150))
        Image.alpha_composite(orig_c, mask_a).save(self.data_dir + "out_features_mask.tif")

    def get_features_map_transparent(self, opacity):
        mask_rgba = np.zeros((self.features.shape[0], self.features.shape[1], 4), dtype=np.uint8)
        mask_rgba[np.where(self.features == 1)] = [250, 0, 0, 0]
        mask_rgba[np.where(self.features == 2)] = [0, 250, 0, 0]
        mask_rgba[np.where(self.features == 3)] = [0, 0, 250, 0]
        mask_rgba[:, :, 3] = opacity
        return Image.fromarray(mask_rgba)

    def get_optical_rgb_with_features_mask(self, opacity=60):
        features_map = self.get_features_map_transparent(opacity)
        orig = Image.fromarray(self.optical_rgb).convert('RGBA')
        return Image.alpha_composite(orig, features_map)

    def get_elevation_with_features_mask(self, opacity=60):
        features_map = self.get_features_map_transparent(opacity)
        orig = Image.fromarray(self.elevation).convert('RGBA')
        return Image.alpha_composite(orig, features_map)


    # to use as input for tensorflow dataset from generator
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
    #loader.get_elevation_with_features_mask()
    #loader.prepare_folders()
    loader.prepare_datasets(output=DataOutput.H5)
    #loader.prepare_valid()
    #loader.prepare_test()
    #loader.prepare_test_with_labels()
    #loader.prepare_all_patches()
    # loader.parse_meta_with_gdal()

