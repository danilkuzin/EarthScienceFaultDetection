import os
import shutil

import gdal
import h5py
import yaml
import numpy as np
from PIL import Image, ImageDraw

from src.DataPreprocessor.region_dataset import FeatureValue, RegionDataset, \
    OutOfBoundsException
from src.config import data_path

from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.DataPreprocessor.DataIOBackend.utm_coord import UtmCoord


# from Philip: "I have checked the kml files, and the points are usually
# within +/-100 m nof the fault but -- as we discussed -- the "context" of
# the fault is broader than this, and I think that at 500-m-wide band is good."
# Since the pixel is 30m, we have a line of width 4 pixels to cover +/-100 m

region_ind = 6
region_data_folder = "Region 7 - Nevada train"
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

with open(
        f'{data_path}/raw_data/{region_data_folder}/'
        f'data_5km_faults/Region_7_fault_lines.utm') as file_object:
    content = file_object.readlines()
    fault_lines = utm_coord.process_content(content)

    # fault_coords = []
    # for line in content[1:]:
    #     regex_matches = line.split(',')
    #     extracted_utm_floats = list(map(float, regex_matches))
    #     if len(extracted_utm_floats) == 5:
    #         pixel_coords = utm_coord.transform_coordinates(
    #             extracted_utm_floats[0], extracted_utm_floats[1])
    #         fault_coords.append(pixel_coords)

with open(
        f'{data_path}/raw_data/{region_data_folder}/'
        f'data_5km_faults/No_Fault_Picks.txt') as file_object:
    content = file_object.readlines()
    non_fault_coords = []
    for line in content[1:]:
        regex_matches = line.split(',')
        extracted_utm_floats = list(map(float, regex_matches))
        if len(extracted_utm_floats) == 5:
            pixel_coords = utm_coord.transform_coordinates(
                extracted_utm_floats[0], extracted_utm_floats[1])
            non_fault_coords.append(pixel_coords)

# debug visualisation
im_np = np.array(gdal.Open(f'{data_path}/raw_data/{region_data_folder}/r.tif',
                 gdal.GA_ReadOnly).ReadAsArray())

im = Image.fromarray(im_np).convert("RGB")
im_width, im_height = im.size

# radius = 100
# for point in fault_coords:
#     bounding_box_for_circle_draw = [point[0]-radius, point[1]-radius,
#                                     point[0]+radius, point[1]+radius]
#     ImageDraw.Draw(im).ellipse(
#         bounding_box_for_circle_draw, fill='cyan', outline='cyan', width=1)
# im.show()

segmentation_mask_np = np.zeros((im_height, im_width), dtype=np.bool)
segmentation_mask = Image.fromarray(segmentation_mask_np)
for line_coord in fault_lines:
    ImageDraw.Draw(segmentation_mask).line(line_coord, fill='white', width=4)
segmentation_mask.show()
segmentation_mask_np = np.array(segmentation_mask)

segmentation_mask_np = segmentation_mask_np.astype(np.int)
segmentation_mask_np[segmentation_mask_np == 0] = \
    FeatureValue.UNDEFINED.value
segmentation_mask_np[segmentation_mask_np == 1] = FeatureValue.FAULT.value
for non_fault_point in non_fault_coords:
    segmentation_mask_np[
        (non_fault_point[1], non_fault_point[0])] = FeatureValue.NONFAULT.value

region_dataset = RegionDataset(region_ind)
output_path = f"{data_path}/train_data/regions_{region_ind}_segmentation_mask/"
if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path)

patch_counter = 0
for line in fault_lines:
    for fault_point_ind in range(len(line)):
        # exclude first and last point of the line to make sure the mask is
        # complete within a subimage

        fault_point = line[fault_point_ind]

        try:
            # we use these coordinates for numpy arrays and fault_point in
            # xy-coordinates
            left_border, right_border, top_border, bottom_border = \
                region_dataset._borders_from_center(
                    (fault_point[1], fault_point[0]),
                    patch_size=(156, 156))
            coords = np.array((left_border, right_border,
                               top_border, bottom_border))
            imgs = region_dataset.concatenate_full_patch(
                left_border, right_border, top_border,
                bottom_border)

            lbls = np.copy(segmentation_mask_np[left_border:right_border,
                           top_border:bottom_border])
            lbls[lbls == FeatureValue.UNDEFINED.value] = FeatureValue.NONFAULT.value
            PLACEHOLDER = 1000
            lbls[lbls == FeatureValue.FAULT.value] = PLACEHOLDER
            lbls[lbls == FeatureValue.NONFAULT.value] = 0
            lbls[lbls == PLACEHOLDER] = 1

            # fault point can be too close to the end of the line such that we
            # cannot be sure with the mask segmentation on the edges of the
            # sugimage, therefore we need to check the edges
            if (np.all(lbls[-2:] == 0) and np.all(lbls[:, -2:] == 0)) or\
                    (np.all(lbls[:2] == 0) and np.all(lbls[:, :2] == 0)):
                # either first/last two columns and rows are fully zeros
                continue

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
                (non_fault_point[1], non_fault_point[0]), patch_size=(156, 156))
        coords = np.array(
            (left_border, right_border, top_border, bottom_border))
        imgs = region_dataset.concatenate_full_patch(
            left_border, right_border, top_border,
            bottom_border)

        lbls = np.copy(segmentation_mask_np[left_border:right_border,
                       top_border:bottom_border])
        lbls[:] = 0

        with h5py.File(f'{output_path}data_{patch_counter}.h5', 'w') as hf:
            hf.create_dataset("img", data=imgs)
            hf.create_dataset("lbl", data=lbls)
            hf.create_dataset("coord", data=coords)

        patch_counter += 1

    except OutOfBoundsException:
        pass







