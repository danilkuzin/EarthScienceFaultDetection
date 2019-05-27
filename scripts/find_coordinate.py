import logging
from typing import Tuple

from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.DataPreprocessor.data_preprocessor import Mode
from src import config


def find_coordinate(region_ind: int, coordinate: Tuple[float, float]) -> Tuple[int, int]:
    data_io_backend = GdalBackend()
    data_io_backend.parse_meta_with_gdal(global_params.data_preprocessor_paths[region_ind] + "/r.tif")

    gt = data_io_backend.gdal_options['geotransform']

    x_pixel = int((coordinate[0] - gt[0]) / gt[1])
    y_pixel = int((coordinate[1] - gt[3]) / gt[5])

    return x_pixel, y_pixel


if __name__ == "__main__":

    logging.basicConfig(level=logging.CRITICAL)

    regions_ind = [3, 6]
    points = [(462082.28, 4319857.87), (520337.87, 4302438.32)]

    for region_ind in regions_ind:
        print(f"region: {region_ind}")
        preprocessor = config.data_preprocessor_generator(Mode.TEST, region_ind)
        for point_ind, point in enumerate(points):
            image_point = find_coordinate(region_ind, point)
            for key, val in preprocessor.channels.items():
                print(f"channel: {key}, value at point {point_ind}: {val[image_point[1], image_point[0]]}")

            for key, val in preprocessor.normalised_channels.items():
                print(f"normalised channel: {key}, value at point {point_ind}: {val[image_point[1], image_point[0]]}")




