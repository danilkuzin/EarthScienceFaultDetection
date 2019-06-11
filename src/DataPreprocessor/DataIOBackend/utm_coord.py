from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from pathlib import Path
from typing import List, Tuple


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
        west_east_pixel_resolution (float): horizontal resolution fo the image.
        top_left_y (float): UTM coordinate of the bottom-right corner of the
            predefined image.
        north_south_pixel_resolution (float): vertical resolution fo the image.
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
            extracted_utm_floats = UtmCoord.extract_floats_from_string(line)
            # 3 below is specific for current input files format
            if len(extracted_utm_floats) == 3:
                pixel_coords = self.transform_coordinates(
                    extracted_utm_floats[0], extracted_utm_floats[1])
                coords.append(pixel_coords)
            if line.startswith('>'):
                if coords:
                    geometries.append(coords)
                    coords = []

        if coords:
            geometries.append(coords)
        return geometries

    def read_geometry(self, files: List[Path]) -> List[List[Tuple[int, int]]]:
        """
        Extract geometrical figures from the list of files.

        Args:
            files ([str]): List of file paths.

        Returns:
            List[List[Tuple[int, int]]]: a list of geometrical figures, where
            each figure is a list of points in image coordinates.

        """
        geometries = []
        for cur_file in files:
            with open(str(cur_file)) as file_object:
                content = file_object.readlines()

            geometries += self.process_content(content)

        return geometries
