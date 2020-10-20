import os
import pathlib
import shutil

import gdal
import h5py
import yaml
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from src.DataPreprocessor.preprocessed_data import PreprocessedData
from src.DataPreprocessor.raw_data_preprocessor import RawDataPreprocessor
from src.DataPreprocessor.region_dataset import FeatureValue, RegionDataset, \
    OutOfBoundsException
from src.config import data_path, data_preprocessor_params

from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.DataPreprocessor.DataIOBackend.utm_coord import UtmCoord


# from Philip: "I have checked the kml files, and the points are usually
# within +/-100 m nof the fault but -- as we discussed -- the "context" of
# the fault is broader than this, and I think that at 500-m-wide band is good."
# Since the pixel is 30m, we have a line of width 4 pixels to cover +/-100 m

def is_point_strictly_inside_box(point, box):
    """

    Args:
        point: [x,y]
        box: [ymin, ymax, xmin, xmax]

    Returns:

    """
    if box[0] < point[1] < box[1]:
        if box[2] < point[0] < box[3]:
            return True

    return False


region_ind = 6
region_data_folder = "Region 7 - Nevada train"
channel_list = ['optical_rgb', 'elevation', 'slope', 'nir', 'topographic_roughness']
input_path = f'{data_path}/labels_from_Philip/Faults/'
output_path = f"{data_path}/train_data/regions_{region_ind}_" \
              f"segmentation_mask_rgb_elev_slope_nir_tri_two_classes/"

front_range_fault_files = ['LQ_Longer_than_5_km_Range_Front.utm',
                           'LLQ_Longer_than_5_km_Range_Front.utm']

data_io_backend = GdalBackend()
with open(
        f"{data_path}/preprocessed/{region_ind}/gdal_params.yaml",
        'r') as stream:
    gdal_params = yaml.safe_load(stream)

data_io_backend.set_params(gdal_params['driver_name'],
                           gdal_params['projection'],
                           eval(gdal_params['geotransform']))


utm_coord = UtmCoord(data_io_backend.geotransform[0],
                     data_io_backend.geotransform[1],
                     data_io_backend.geotransform[3],
                     data_io_backend.geotransform[5])

# read front range fault
front_range_fault_lines = []
for file in front_range_fault_files:
    with open(f'{input_path}/{file}') as file_object:
        content = file_object.readlines()
        current_lines = utm_coord.process_content(content)
        front_range_fault_lines += current_lines

# debug visualisation
im_np = np.array(gdal.Open(f'{data_path}/raw_data/{region_data_folder}/r.tif',
                 gdal.GA_ReadOnly).ReadAsArray())

im = Image.fromarray(im_np).convert("RGB")
im_width, im_height = im.size

im_np = np.array(im).astype(np.uint8)

empty_placeholder = np.zeros((im_height, im_width), dtype=np.bool)
segmentation_mask = Image.fromarray(empty_placeholder)
for ind, line_coord in enumerate(front_range_fault_lines):
    ImageDraw.Draw(segmentation_mask).line(line_coord, fill='white', width=4)
segmentation_front_range_mask_np = np.array(segmentation_mask)


segmentation_mask_np = FeatureValue.UNDEFINED.value * np.ones(
    (im_height, im_width), dtype=np.int)
segmentation_mask_np[segmentation_front_range_mask_np == 1] = \
    FeatureValue.FAULT.value


region_dataset = RegionDataset(region_ind)

# preprocessed_data = PreprocessedData(region_id=region_ind)
# preprocessed_data.load()
# input_image_rgb_np = preprocessed_data.channels['optical_rgb']

patch_counter = 0
patch_number = 2
should_break = False
lines = front_range_fault_lines
for ind, line in enumerate(lines):
    edge_one = line[0]
    try:
        left_border, right_border, top_border, bottom_border = \
            region_dataset._borders_from_center(
                (edge_one[1], edge_one[0]),
                patch_size=(156, 156))
        edge_one_box = np.array((left_border, right_border,
                                 top_border, bottom_border))
    except OutOfBoundsException:
        edge_one_box = None
    edge_two = line[-1]
    try:
        left_border, right_border, top_border, bottom_border = \
            region_dataset._borders_from_center(
                (edge_two[1], edge_two[0]),
                patch_size=(156, 156))
        edge_two_box = np.array((left_border, right_border,
                                 top_border, bottom_border))
    except OutOfBoundsException:
        edge_two_box = None
    for fault_point_ind in range(len(line)):
        # exclude first and last point of the line to make sure the mask is
        # complete within a subimage

        fault_point = line[fault_point_ind]

        # fault point can be too close to the end of the line such that we
        # cannot be sure with the mask segmentation on the edges of the
        # sugimage, therefore we need to check the edges:
        # valid fault point should be outside of patch size box around both
        # edges

        if ((edge_one_box is not None and
             is_point_strictly_inside_box(fault_point, edge_one_box)) or
                (edge_two_box is not None and
                 is_point_strictly_inside_box(fault_point, edge_two_box))):
            continue

        try:
            # we use these coordinates for numpy arrays and fault_point in
            # xy-coordinates
            left_border, right_border, top_border, bottom_border = \
                region_dataset._borders_from_center(
                    (fault_point[1], fault_point[0]),
                    patch_size=(156, 156))
            coords = np.array((left_border, right_border,
                               top_border, bottom_border))

            # img = input_image_rgb_np[left_border:right_border, top_border:bottom_border, :]
            img = region_dataset.concatenate_full_patch(
                left_border, right_border, top_border,
                bottom_border, channel_list=channel_list)

            lbls = np.copy(segmentation_mask_np[left_border:right_border,
                           top_border:bottom_border])
            lbls[lbls == FeatureValue.UNDEFINED.value] = FeatureValue.NONFAULT.value
            PLACEHOLDER = 1000
            lbls[lbls == FeatureValue.FAULT.value] = PLACEHOLDER
            lbls[lbls == FeatureValue.NONFAULT.value] = 0
            lbls[lbls == PLACEHOLDER] = 1
            lbls[lbls == FeatureValue.BASIN_FAULT.value] = 0

            plt.imshow(img[:, :, 3])
            plt.axis('off')
            plt.savefig(f"elevation_image_patch_{patch_counter}.png", bbox_inches='tight')
            plt.show()

            plt.imshow(lbls)
            plt.axis('off')
            plt.savefig(f"front_fault_mask_{patch_counter}.png",
                        bbox_inches='tight')
            plt.show()

            patch_counter += 1

            break

        except OutOfBoundsException:
            pass

    if patch_counter >= patch_number:
        should_break = True
        break










