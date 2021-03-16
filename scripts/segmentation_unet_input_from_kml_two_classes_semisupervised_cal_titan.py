import os
import random
import shutil

from osgeo import gdal
import geopandas
import fiona
import rasterio
import rasterio.warp
import shapely.geometry
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


region_ind = 12
region_data_folder = "Region 12 - Nothern California"
channel_list = ['optical_rgb', 'elevation', 'nir', 'topographic_roughness',
                'flow', 'sar1', 'sar2']
input_path = f"{data_path}/raw_data/{region_data_folder}"
output_path = f"{data_path}/train_data/" \
              f"regions_{region_ind}_segmentation_mask/"

fault_files = ["HAZMAP.kml"]
fiona.drvsupport.supported_drivers['libkml'] = 'rw'
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'

non_fault_files = ['NotFault_polygons_NorCal_v2.kml']

points_per_non_fault_polygon = 50

data_io_backend = GdalBackend()
with open(
        f"/mnt/data/datasets/DataForEarthScienceFaultDetection/"
        f"preprocessed/{region_ind}/gdal_params.yaml",
        'r') as stream:
    gdal_params = yaml.safe_load(stream)

data_io_backend.set_params(gdal_params['driver_name'],
                           gdal_params['projection'],
                           eval(gdal_params['geotransform']))


utm_coord = UtmCoord(data_io_backend.geotransform[0],
                     data_io_backend.geotransform[1],
                     data_io_backend.geotransform[3],
                     data_io_backend.geotransform[5])

dataset = rasterio.open(f'/mnt/data/datasets/'
                        f'DataForEarthScienceFaultDetection/raw_data/'
                        f'{region_data_folder}/r_landsat.tif')
latlon_bounds = rasterio.warp.transform_bounds(
    dataset.crs, "EPSG:4326", *dataset.bounds)

# read faults
strike_slip_fault_lines = []
thrust_fault_lines = []
for file in fault_files:
    data = geopandas.read_file(f'{input_path}/{file}')
    filtered_data = data[data.intersects(shapely.geometry.box(*latlon_bounds))]

    filtered_data = filtered_data.to_crs(dataset.crs)

    for index in range(filtered_data.shape[0]):
        fault_line_data = filtered_data.iloc[index]
        if fault_line_data['geometry'].type == 'MultiLineString':
            for line in fault_line_data['geometry']:
                utm_coord_list = list(line.coords)
                coords = []
                for point_utm in utm_coord_list:
                    pixel_coords = utm_coord.transform_coordinates(
                        point_utm[0], point_utm[1])
                    coords.append(pixel_coords)
                if fault_line_data['disp_slip_'] == 'strike slip':
                    strike_slip_fault_lines.append(coords)
                elif fault_line_data['disp_slip_'] == 'thrust':
                    thrust_fault_lines.append(coords)
                else:
                    print('UNEXPECTED FAULT TYPE!')
        else:
            utm_coord_list = list(fault_line_data['geometry'].coords)
            coords = []
            for point_utm in utm_coord_list:
                pixel_coords = utm_coord.transform_coordinates(
                    point_utm[0], point_utm[1])
                coords.append(pixel_coords)
            if fault_line_data['disp_slip_'] == 'strike slip':
                strike_slip_fault_lines.append(coords)
            elif fault_line_data['disp_slip_'] == 'thrust':
                thrust_fault_lines.append(coords)
            else:
                print('UNEXPECTED FAULT TYPE!')


# read non-faults
non_fault_coords = []
non_fault_polygons = []
for file in non_fault_files:
    data = geopandas.read_file(f'{input_path}/{file}')
    data = data.to_crs(dataset.crs)

    for index in range(data.shape[0]):
        non_fault_data = data.iloc[index]
        if non_fault_data['geometry'].type == 'MultiPolygon':
            for polygon in non_fault_data['geometry']:
                utm_coord_list = list(polygon.exterior.coords)
                coords = []
                for point_utm in utm_coord_list:
                    pixel_coords = utm_coord.transform_coordinates(
                        point_utm[0], point_utm[1])
                    coords.append(pixel_coords)
                non_fault_polygons.append(coords)

                (minx, miny, maxx, maxy) = polygon.bounds
                for counter in range(points_per_non_fault_polygon):
                    sampled = False
                    while not sampled:
                        sample_x = random.uniform(minx, maxx)
                        sample_y = random.uniform(miny, maxy)

                        sample_point = shapely.geometry.Point(sample_x, sample_y)
                        if sample_point.within(polygon):
                            sampled = True

                    pixel_coords = utm_coord.transform_coordinates(
                        sample_x, sample_y)
                    non_fault_coords.append(pixel_coords)

        else:
            print('UNEXPECTED GEOMETRY TYPE FOR NON FAULTS!')

    # with open(f'{input_path}/{file}') as file_object:
    #     content = file_object.readlines()
    #     for line in content[3:]:
    #         extracted_utm_floats = utm_coord.extract_floats_from_string(line)
    #         if len(extracted_utm_floats) > 0:
    #             pixel_coords = utm_coord.transform_coordinates(
    #                 extracted_utm_floats[0], extracted_utm_floats[1])
    #             non_fault_coords.append(pixel_coords)

# debug visualisation
im_np = np.array(gdal.Open(f'/mnt/data/datasets/'
                           f'DataForEarthScienceFaultDetection/raw_data/'
                           f'{region_data_folder}/r_landsat.tif',
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

land_mask_np = np.array(
    gdal.Open(f'/mnt/data/datasets/'
              f'DataForEarthScienceFaultDetection/raw_data/'
              f'{region_data_folder}/land_mask.tif',
                 gdal.GA_ReadOnly).ReadAsArray()).astype(bool)

empty_placeholder = np.zeros((im_height, im_width), dtype=bool)
segmentation_mask = Image.fromarray(empty_placeholder)
for ind, line_coord in enumerate(strike_slip_fault_lines):
    ImageDraw.Draw(segmentation_mask).line(line_coord, fill='white', width=4)
segmentation_strike_slip_mask_np = np.array(segmentation_mask)
segmentation_strike_slip_mask_np = np.logical_and(
    segmentation_strike_slip_mask_np, land_mask_np)
# for ind, line_coord in enumerate(strike_slip_fault_lines):
#     ImageDraw.Draw(segmentation_mask).line(line_coord, fill='white', width=150)
# segmentation_strike_slip_non_fault_mask_np = np.array(segmentation_mask)

segmentation_mask = Image.fromarray(empty_placeholder)
for ind, line_coord in enumerate(thrust_fault_lines):
    ImageDraw.Draw(segmentation_mask).line(line_coord, fill='white', width=4)
segmentation_thrust_mask_np = np.array(segmentation_mask)
segmentation_thrust_mask_np = np.logical_and(
    segmentation_thrust_mask_np, land_mask_np)
# for ind, line_coord in enumerate(thrust_fault_lines):
#     ImageDraw.Draw(segmentation_mask).line(line_coord, fill='white', width=150)
# segmentation_thrust_non_fault_mask_np = np.array(segmentation_mask)

segmentation_mask = Image.fromarray(empty_placeholder)
# for ind, non_fault_point in enumerate(non_fault_coords):
#     ImageDraw.Draw(segmentation_mask).rectangle(
#         [int(non_fault_point[0]-140/2), int(non_fault_point[1]-140/2),
#          int(non_fault_point[0]+140/2), int(non_fault_point[1]+140/2)],
#         fill='white')
for ind, non_fault_polygon in enumerate(non_fault_polygons):
    # skip last point as it is the first one and ImageDraw connects
    # the last and the first points automatically
    ImageDraw.Draw(segmentation_mask).polygon(non_fault_polygon[:-1],
                                              fill='white')
segmentation_non_fault_mask_np = np.array(segmentation_mask)

segmentation_mask_np = FeatureValue.UNDEFINED.value * np.ones(
    (im_height, im_width), dtype=np.int)
segmentation_mask_np[segmentation_non_fault_mask_np == 1] = \
    FeatureValue.NONFAULT.value
# segmentation_mask_np[segmentation_strike_slip_non_fault_mask_np == 1] = \
#     FeatureValue.NONFAULT.value
# segmentation_mask_np[segmentation_thrust_non_fault_mask_np == 1] = \
#     FeatureValue.NONFAULT.value
segmentation_mask_np[segmentation_strike_slip_mask_np == 1] = \
    FeatureValue.STRIKE_SLIP_FAULT.value
segmentation_mask_np[segmentation_thrust_mask_np == 1] = \
    FeatureValue.THRUST_FAULT.value


segmentation_mask_np_vis = np.zeros((im_height, im_width, 3), dtype=np.uint8)
segmentation_mask_np_vis[segmentation_mask_np == FeatureValue.STRIKE_SLIP_FAULT.value, 0] = 255
segmentation_mask_np_vis[segmentation_mask_np == FeatureValue.THRUST_FAULT.value, 1] = 255
segmentation_mask_np_vis[segmentation_mask_np == FeatureValue.NONFAULT.value, 2] = 255
vis_mask = Image.fromarray(segmentation_mask_np_vis)
# vis_mask.show()
# data_io_backend.write_image('train_data_vis.tif', segmentation_mask_np_vis)

region_dataset = RegionDataset(region_ind)
if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path)

patch_counter = 0
lines = strike_slip_fault_lines + thrust_fault_lines
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
            lbls[lbls == FeatureValue.STRIKE_SLIP_FAULT.value] = PLACEHOLDER
            lbls[lbls == FeatureValue.NONFAULT.value] = 0
            lbls[lbls == PLACEHOLDER] = 1
            lbls[lbls == FeatureValue.THRUST_FAULT.value] = 2
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

        PLACEHOLDER = 1000
        lbls[lbls == FeatureValue.STRIKE_SLIP_FAULT.value] = PLACEHOLDER
        lbls[lbls == FeatureValue.NONFAULT.value] = 0
        lbls[lbls == PLACEHOLDER] = 1
        lbls[lbls == FeatureValue.THRUST_FAULT.value] = 2
        lbls[lbls == FeatureValue.UNDEFINED.value] = 3

        with h5py.File(f'{output_path}data_{patch_counter}.h5', 'w') as hf:
            hf.create_dataset("img", data=imgs)
            hf.create_dataset("lbl", data=lbls)
            hf.create_dataset("coord", data=coords)

        patch_counter += 1

    except OutOfBoundsException:
        pass







