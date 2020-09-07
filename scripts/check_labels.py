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
channel_list = ['optical_rgb', 'elevation', 'slope']
input_folder = f'{data_path}/labels_from_Philip/Faults/'


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
        f'{input_folder}LQ_Piedmont_and_Basins.utm') as file_object:
    content = file_object.readlines()
    fault_lines_1 = utm_coord.process_content(content)

    # fault_coords = []
    # for line in content[1:]:
    #     regex_matches = line.split(',')
    #     extracted_utm_floats = list(map(float, regex_matches))
    #     if len(extracted_utm_floats) == 5:
    #         pixel_coords = utm_coord.transform_coordinates(
    #             extracted_utm_floats[0], extracted_utm_floats[1])
    #         fault_coords.append(pixel_coords)

with open(
        f'{input_folder}LLQ_Piedmont_and_Basins.utm') as file_object:
    content = file_object.readlines()
    fault_lines_2 = utm_coord.process_content(content)

fault_lines = fault_lines_1 + fault_lines_2

with open(
        f'{input_folder}No_Faults_June.utm') as file_object:
    content = file_object.readlines()
    non_fault_coords = []
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
im.show()

im_np = np.array(im).astype(np.uint8)

data_io_backend.write_image('test_points.tif', im_np)

segmentation_mask_np = np.zeros((im_height, im_width), dtype=np.bool)
segmentation_mask = Image.fromarray(segmentation_mask_np)
for ind, line_coord in enumerate(fault_lines):
    ImageDraw.Draw(segmentation_mask).line(line_coord, fill='white', width=4)
segmentation_mask.show()

segmentation_mask_np = (np.array(segmentation_mask) * 255).astype(np.uint8)

data_io_backend.write_image('test.tif', segmentation_mask_np)

