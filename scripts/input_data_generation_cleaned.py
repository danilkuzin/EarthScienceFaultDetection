import os
import shutil
import re

import gdal # sometimes it should be "from osgeo import gdal"
import h5py
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Tuple
from enum import Enum


class OutOfBoundsException(Exception):
    pass


class FeatureValue(Enum):
    UNDEFINED = -1
    FAULT = 0
    FAULT_LOOKALIKE = 1
    NONFAULT = 2
    BASIN_FAULT = 3
    STRIKE_SLIP_FAULT = 4
    THRUST_FAULT = 5


# class to read utm coordinates and convert them to pixel coordinates
class UtmCoord(object):
    """
    Extract different geometric objects from the custom format representing UTM
    coordinates.

    Converts coordinates into image coordinates of the predefined image.
    The parameters of the predefined image correspond to the subset of
    geotransform properties of the GeoTIFF image.

    Attributes:
        top_left_x (float): UTM coordinate of the top-left corner of the
            predefined image.
        west_east_pixel_resolution (float): horizontal resolution of the image.
        top_left_y (float): UTM coordinate of the bottom-right corner of the
            predefined image.
        north_south_pixel_resolution (float): vertical resolution of the image.
    """

    def __init__(self, top_left_x: float, west_east_pixel_resolution: float,
                 top_left_y: float, north_south_pixel_resolution: float):
        self.top_left_x = top_left_x
        self.west_east_pixel_resolution = west_east_pixel_resolution
        self.top_left_y = top_left_y
        self.north_south_pixel_resolution = north_south_pixel_resolution

    def transform_coordinates(self, utm_coord_x: float,
                              utm_coord_y: float) -> Tuple[int, int]:
        """
        Transforms UTM coordinates into image coordinates.

        Args:
            utm_coord_x (float): UTM x coordinate.
            utm_coord_y (float): UTM y coordinate.

        Returns:
            Tuple[int, int] of image coordinates.

        """
        x_pixel = int(
            (utm_coord_x - self.top_left_x) / self.west_east_pixel_resolution)
        y_pixel = int((utm_coord_y - self.top_left_y) /
                      self.north_south_pixel_resolution)
        return x_pixel, y_pixel

    @staticmethod
    def extract_floats_from_string(input_str: str) -> List[float]:
        """
        Extract list of float numbers from string without a carriage return.

        Args:
            input_str (str): Input string.

        Returns:
            List[float] of floats.

        """
        regex_matches = re.findall(
            r'[-+]?[.]?[\d]+(?:,\d\d\d)*[.]?\d*(?:[eE][-+]?\d+)?', input_str)

        return list(map(float, regex_matches))

    def process_content(self,
                        content: List[str]) -> List[List[Tuple[int, int]]]:
        """
        Process the predefined list of UTM coordinate strings into
        a list of geometrical figures in image coordinates.

        Args:
            content ([str]): Input of the form
                ['>\n',
                 '           x_1_1          y_1_1       t\n',
                 '           x_1_2          y_1_2       t\n',
                 '>\n',
                 '           x_2_1          y_2_1      t\n'],
                where x_[i]_[j] is the [j]th float x coordinate of the [i]th
                object, y_[i]_[j] - corresponding y coordinate, t - unused number.

        Returns:
            List[List[Tuple[int, int]]]: a list of geometrical figures, where
            each figure is a list of points in image coordinates.

        """
        geometries = []
        coords = []
        for line in content:
            if line.startswith('>'):
                if coords:
                    geometries.append(coords)
                    coords = []
                    continue
            extracted_utm_floats = UtmCoord.extract_floats_from_string(line)
            # 3 and 5 below are specific for current input files format
            if len(extracted_utm_floats) == 3 or len(extracted_utm_floats) == 5:
                pixel_coords = self.transform_coordinates(
                    extracted_utm_floats[0], extracted_utm_floats[1])
                coords.append(pixel_coords)

        if coords:
            geometries.append(coords)
        return geometries


# class to upload and deal with normalised input data
class RegionDataset:
    def __init__(self, path: str):
        self.channels = dict(elevation=None)
        self.load(path)

    def load(self, path):
        # load from normalised data
        with h5py.File(f"{path}/data.h5", 'r') as hf:
            optical_r = hf["optical_r"][:]
            optical_g = hf["optical_g"][:]
            optical_b = hf["optical_b"][:]
            self.channels['optical_rgb'] = np.stack(
                (optical_r, optical_g, optical_b), axis=-1)
            self.channels['elevation'] = hf["elevation"][:]
            if "slope" in hf:
                self.channels['slope'] = hf["slope"][:]
            else:
                self.channels['slope'] = None
            if "nir" in hf:
                self.channels['nir'] = hf["nir"][:]
            else:
                self.channels['nir'] = None
            if "topographic_roughness" in hf:
                self.channels['topographic_roughness'] = \
                    hf["topographic_roughness"][:]
            else:
                self.channels['topographic_roughness'] = None
            if "ultrablue" in hf:
                self.channels['ultrablue'] = hf["ultrablue"][:]
            else:
                self.channels['ultrablue'] = None
            if "swir1" in hf:
                self.channels['swir1'] = hf["swir1"][:]
            else:
                self.channels['swir1'] = None
            if "swir2" in hf:
                self.channels['swir2'] = hf["swir2"][:]
            else:
                self.channels['swir2'] = None
            if "flow" in hf:
                self.channels['flow'] = hf["flow"][:]
            else:
                self.channels['flow'] = None
            if "erosion" in hf:
                self.channels['erosion'] = hf["erosion"][:]
            else:
                self.channels['erosion'] = None
            if "sar1" in hf:
                self.channels['sar1'] = hf["sar1"][:]
            else:
                self.channels['sar1'] = None
            if "sar1" in hf:
                self.channels['sar2'] = hf["sar2"][:]
            else:
                self.channels['sar2'] = None
            if "incision" in hf:
                self.channels['incision'] = hf["incision"][:]
            else:
                self.channels['incision'] = None
            # features would be empty always, legacy part
            self.features = hf["features"][:]

    def get_data_shape(self):
        return self.channels['elevation'].shape[0], \
               self.channels['elevation'].shape[1], len(self.channels)

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

    def concatenate_full_patch(
            self, left_border: int, right_border: int, top_border: int,
            bottom_border: int, channel_list: List[str]):
        np_channel_data = []
        for channel in channel_list:
            if self.channels[channel].ndim == 3:
                np_channel_data.append(self.channels[channel][
                                       left_border:right_border,
                                       top_border:bottom_border])
            else:
                np_channel_data.append(np.expand_dims(
                    self.channels[channel][
                        left_border:right_border, top_border:bottom_border],
                    axis=2))
        return np.concatenate(np_channel_data, axis=2)


region_ind = 6
data_path = 'output_data' # PLACE PATH TO YOUR DATA FOLDER HERE
region_data_folder = "Region 7 - Nevada train"
channel_list = ['elevation']
output_path = f"{data_path}/" \
              f"train_data/" \
              f"segmentation_masks/"

front_range_fault_files = ['Region_6_Fault_Picks.utm']

# the first file is repeated 3 times to enforce the model not to pick up
# some of the look-alike as faults, however, it was done some time ago,
# we haven't tested whether it is required in the latest experiments
non_fault_files = ['RTW_Not_Faults_Edited.utm',
                   'RTW_Not_Faults_Edited.utm',
                   'RTW_Not_Faults_Edited.utm',
                   'No_Faults_June.utm']


r_channel_path = f'{data_path}/{region_data_folder}/r_landsat.tif'
dataset = gdal.Open(r_channel_path, gdal.GA_ReadOnly)
if not dataset:
    raise FileNotFoundError(dataset)


gdal_params = {}
gdal_params['driver_name'] = dataset.GetDriver().ShortName
gdal_params['projection'] = dataset.GetProjection()
gdal_params['geotransform'] = dataset.GetGeoTransform()


utm_coord = UtmCoord(gdal_params['geotransform'][0],
                     gdal_params['geotransform'][1],
                     gdal_params['geotransform'][3],
                     gdal_params['geotransform'][5])

# read front range fault
front_range_fault_lines = []
for file in front_range_fault_files:
    with open(f'{data_path}/{file}') as file_object:
        content = file_object.readlines()
        current_lines = utm_coord.process_content(content)
        front_range_fault_lines += current_lines

# read non-faults
non_fault_coords = []
for file in non_fault_files:
    with open(f'{data_path}/{file}') as file_object:
        content = file_object.readlines()
        for line in content[2:]:
            extracted_utm_floats = utm_coord.extract_floats_from_string(line)
            if len(extracted_utm_floats) > 0:
                pixel_coords = utm_coord.transform_coordinates(
                    extracted_utm_floats[0], extracted_utm_floats[1])
                non_fault_coords.append(pixel_coords)

# debug visualisation
im_np = np.array(dataset.ReadAsArray())

im = Image.fromarray(im_np).convert("RGB")
im_width, im_height = im.size

radius = 10
for point in non_fault_coords:
    bounding_box_for_circle_draw = [point[0]-radius, point[1]-radius,
                                    point[0]+radius, point[1]+radius]
    ImageDraw.Draw(im).ellipse(
        bounding_box_for_circle_draw, fill='orange', outline='orange', width=1)
im.show()


# Create labels:
# On the whole image (region 6, Train Nevada):
# - Lines for faults labels are drawn with width of 150 pixels (~4.5km).
# Pixels of these lines are labelled as non-faults.
# - Lines for fault labels are drawn also with width of 4 pixels (~120 meters).
# These pixels are labelled as faults (overwriting the previous labels
# as non-faults).
# - Around points of non-faults labels a rectangle of the size of 140 pixels
# (~4.2km) are drawn. These pixels are labelled
# - All remaining pixels are labelled as unassigned. Unassigned labels
# shouldn't be counted in the loss function.
empty_placeholder = np.zeros((im_height, im_width), dtype=np.bool)
segmentation_mask = Image.fromarray(empty_placeholder)
for ind, line_coord in enumerate(front_range_fault_lines):
    ImageDraw.Draw(segmentation_mask).line(line_coord, fill='white', width=4)
segmentation_front_range_mask_np = np.array(segmentation_mask)
for ind, line_coord in enumerate(front_range_fault_lines):
    ImageDraw.Draw(segmentation_mask).line(line_coord, fill='white', width=150)
segmentation_front_range_non_fault_mask_np = np.array(segmentation_mask)

segmentation_mask = Image.fromarray(empty_placeholder)
for ind, non_fault_point in enumerate(non_fault_coords):
    ImageDraw.Draw(segmentation_mask).rectangle(
        [int(non_fault_point[0]-140/2), int(non_fault_point[1]-140/2),
         int(non_fault_point[0]+140/2), int(non_fault_point[1]+140/2)],
        fill='white')
segmentation_non_fault_mask_np = np.array(segmentation_mask)

segmentation_mask_np = FeatureValue.UNDEFINED.value * np.ones(
    (im_height, im_width), dtype=np.int)
segmentation_mask_np[segmentation_non_fault_mask_np == 1] = \
    FeatureValue.NONFAULT.value
segmentation_mask_np[segmentation_front_range_non_fault_mask_np == 1] = \
    FeatureValue.NONFAULT.value
segmentation_mask_np[segmentation_front_range_mask_np == 1] = \
    FeatureValue.FAULT.value


segmentation_mask_np_vis = np.zeros((im_height, im_width, 3), dtype=np.uint8)
segmentation_mask_np_vis[
    segmentation_mask_np == FeatureValue.FAULT.value, 0] = 255
segmentation_mask_np_vis[
    segmentation_mask_np == FeatureValue.NONFAULT.value, 2] = 255
vis_mask = Image.fromarray(segmentation_mask_np_vis)
vis_mask.show()

# prepare to create image patches from input data
normalised_data_path = f'{data_path}/normalised/{region_ind}'
region_dataset = RegionDataset(normalised_data_path)
if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path)


# Create image patches for training
patch_counter = 0
for ind, line in enumerate(front_range_fault_lines):
    for fault_point_ind in range(len(line)):

        fault_point = line[fault_point_ind]

        try:
            # we use these coordinates for numpy arrays and fault_point in
            # xy-coordinates
            left_border, right_border, top_border, bottom_border = \
                region_dataset._borders_from_center(
                    (fault_point[1], fault_point[0]),
                    patch_size=(736, 736))
            coords = np.array((left_border, right_border,
                               top_border, bottom_border))

            imgs = region_dataset.concatenate_full_patch(
                left_border, right_border, top_border,
                bottom_border, channel_list=channel_list)

            lbls = np.copy(segmentation_mask_np[left_border:right_border,
                           top_border:bottom_border])
            # recode labels to be consecutive from 0: 0, 1, 2
            PLACEHOLDER = 1000
            lbls[lbls == FeatureValue.FAULT.value] = PLACEHOLDER
            lbls[lbls == FeatureValue.NONFAULT.value] = 0
            lbls[lbls == PLACEHOLDER] = 1
            lbls[lbls == FeatureValue.UNDEFINED.value] = 2

            with h5py.File(f'{output_path}data_{patch_counter}.h5', 'w') as hf:
                hf.create_dataset("img", data=imgs,
                                  # if to keep the next line, the resulting
                                  # files are smaller, but reading time from
                                  # them is longer. If the opposite effect is
                                  # desired, remove the next line
                                  compression='gzip', compression_opts=9)
                hf.create_dataset("lbl", data=lbls,
                                  compression='gzip', compression_opts=9)
                hf.create_dataset("coord", data=coords,
                                  compression='gzip', compression_opts=9)

            patch_counter += 1

        except OutOfBoundsException:
            pass

for non_fault_point in non_fault_coords:
    try:
        left_border, right_border, top_border, bottom_border = \
            region_dataset._borders_from_center(
                (non_fault_point[1], non_fault_point[0]), patch_size=(736, 736))
        coords = np.array(
            (left_border, right_border, top_border, bottom_border))
        imgs = region_dataset.concatenate_full_patch(
            left_border, right_border, top_border,
            bottom_border, channel_list=channel_list)

        lbls = np.copy(segmentation_mask_np[left_border:right_border,
                       top_border:bottom_border])

        # recode labels to be consecutive from 0: 0, 1, 2
        lbls[lbls == FeatureValue.NONFAULT.value] = 0
        lbls[lbls == FeatureValue.UNDEFINED.value] = 2

        with h5py.File(f'{output_path}data_{patch_counter}.h5', 'w') as hf:
            hf.create_dataset("img", data=imgs,
                              compression='gzip', compression_opts=9)
            hf.create_dataset("lbl", data=lbls,
                              compression='gzip', compression_opts=9)
            hf.create_dataset("coord", data=coords,
                              compression='gzip', compression_opts=9)

        patch_counter += 1

    except OutOfBoundsException:
        pass







