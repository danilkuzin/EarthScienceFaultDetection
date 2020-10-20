import os
import shutil

import gdal
import h5py
import yaml
import numpy as np
from PIL import Image, ImageDraw

import sys
sys.path.extend(['/home/oi260/github/EarthScienceFaultDetection/'])

from src.DataPreprocessor.region_dataset import FeatureValue, RegionDataset, \
    OutOfBoundsException
from src.config import data_path

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
              f"segmentation_mask_rgb_elev_slope_nir_tri_two_classes_" \
              f"semisupervised/"

front_range_fault_files = ['LQ_Longer_than_5_km_Range_Front.utm',
                           'LLQ_Longer_than_5_km_Range_Front.utm']
basin_fault_files = ['LQ_Piedmont_and_Basins.utm',
                     'LLQ_Piedmont_and_Basins.utm']
non_fault_files = ['RTW_Not_Faults_Edited.utm',
                   'RTW_Not_Faults_Edited.utm',
                   'RTW_Not_Faults_Edited.utm',
                   'No_Faults_June.utm']

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

# read basin fault
basin_fault_lines = []
for file in basin_fault_files:
    with open(f'{input_path}/{file}') as file_object:
        content = file_object.readlines()
        current_lines = utm_coord.process_content(content)
        basin_fault_lines += current_lines

# read non-faults
non_fault_coords = []
for file in non_fault_files:
    with open(f'{input_path}/{file}') as file_object:
        content = file_object.readlines()
        for line in content[2:]:
            extracted_utm_floats = utm_coord.extract_floats_from_string(line)
            if len(extracted_utm_floats) > 0:
                pixel_coords = utm_coord.transform_coordinates(
                    extracted_utm_floats[0], extracted_utm_floats[1])
                non_fault_coords.append(pixel_coords)

# debug visualisation
im_np = np.array(gdal.Open(f'{data_path}/raw_data/{region_data_folder}/r.tif',
                 gdal.GA_ReadOnly).ReadAsArray())

im = Image.fromarray(im_np).convert("RGB")
im_width, im_height = im.size

radius = 10
for point in non_fault_coords:
    bounding_box_for_circle_draw = [point[0]-radius, point[1]-radius,
                                    point[0]+radius, point[1]+radius]
    ImageDraw.Draw(im).ellipse(
        bounding_box_for_circle_draw, fill='orange', outline='orange', width=1)
# im.show()

im_np = np.array(im).astype(np.uint8)

# data_io_backend.write_image('test_points.tif', im_np)

empty_placeholder = np.zeros((im_height, im_width), dtype=np.bool)
segmentation_mask = Image.fromarray(empty_placeholder)
for ind, line_coord in enumerate(front_range_fault_lines):
    ImageDraw.Draw(segmentation_mask).line(line_coord, fill='white', width=4)
segmentation_front_range_mask_np = np.array(segmentation_mask)
for ind, line_coord in enumerate(front_range_fault_lines):
    ImageDraw.Draw(segmentation_mask).line(line_coord, fill='white', width=150)
segmentation_front_range_non_fault_mask_np = np.array(segmentation_mask)

segmentation_mask = Image.fromarray(empty_placeholder)
for ind, line_coord in enumerate(basin_fault_lines):
    ImageDraw.Draw(segmentation_mask).line(line_coord, fill='white', width=4)
segmentation_basin_mask_np = np.array(segmentation_mask)
for ind, line_coord in enumerate(basin_fault_lines):
    ImageDraw.Draw(segmentation_mask).line(line_coord, fill='white', width=150)
segmentation_basin_non_fault_mask_np = np.array(segmentation_mask)

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
segmentation_mask_np[segmentation_basin_non_fault_mask_np == 1] = \
    FeatureValue.NONFAULT.value
segmentation_mask_np[segmentation_front_range_mask_np == 1] = \
    FeatureValue.FAULT.value
segmentation_mask_np[segmentation_basin_mask_np == 1] = \
    FeatureValue.BASIN_FAULT.value



segmentation_mask_np_vis = np.zeros((im_height, im_width, 3), dtype=np.uint8)
segmentation_mask_np_vis[segmentation_mask_np == FeatureValue.FAULT.value, 0] = 255
segmentation_mask_np_vis[segmentation_mask_np == FeatureValue.BASIN_FAULT.value, 1] = 255
segmentation_mask_np_vis[segmentation_mask_np == FeatureValue.NONFAULT.value, 2] = 255
vis_mask = Image.fromarray(segmentation_mask_np_vis)
# vis_mask.show()
# data_io_backend.write_image('train_data_vis.tif', segmentation_mask_np_vis)

region_dataset = RegionDataset(region_ind)
if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path)

patch_counter = 0
lines = front_range_fault_lines + basin_fault_lines
for ind, line in enumerate(lines):
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
            PLACEHOLDER = 1000
            lbls[lbls == FeatureValue.FAULT.value] = PLACEHOLDER
            lbls[lbls == FeatureValue.NONFAULT.value] = 0
            lbls[lbls == PLACEHOLDER] = 1
            lbls[lbls == FeatureValue.BASIN_FAULT.value] = 2
            lbls[lbls == FeatureValue.UNDEFINED.value] = 3

            with h5py.File(f'{output_path}data_{patch_counter}.h5', 'w') as hf:
                hf.create_dataset("img", data=imgs)
                hf.create_dataset("lbl", data=lbls)
                hf.create_dataset("coord", data=coords)

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

        lbls[lbls == FeatureValue.NONFAULT.value] = 0
        lbls[lbls == FeatureValue.UNDEFINED.value] = 3

        with h5py.File(f'{output_path}data_{patch_counter}.h5', 'w') as hf:
            hf.create_dataset("img", data=imgs)
            hf.create_dataset("lbl", data=lbls)
            hf.create_dataset("coord", data=coords)

        patch_counter += 1

    except OutOfBoundsException:
        pass







