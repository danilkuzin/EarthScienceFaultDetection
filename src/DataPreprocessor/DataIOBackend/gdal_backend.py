from PIL import Image, ImageDraw
from matplotlib.path import Path

import re
from pathlib import Path
from tempfile import NamedTemporaryFile

from src.DataPreprocessor.DataIOBackend.backend import DataIOBackend
import numpy as np
from osgeo import gdal
import logging

import matplotlib.pyplot as plt

from src.DataPreprocessor.DataIOBackend.utm_coord import UtmCoord
from src.DataPreprocessor.raw_data_preprocessor import FeatureValue


class GdalBackend(DataIOBackend):
    def __init__(self):
        self.driver_name = None
        self.projection = None
        self.geotransform = None

    def __load_1d_raster(self, path: str) -> np.array:
        dataset = gdal.Open(path, gdal.GA_ReadOnly)
        if not dataset:
            raise FileNotFoundError(path)
        return np.array(dataset.ReadAsArray())

    def __load_1d_uint16(self, path: str) -> np.array:
        opt_string = '-ot Byte -of GTiff -scale 0 65535 0 255'
        dataset = gdal.Translate(NamedTemporaryFile(delete=False).name,
                                 gdal.Open(path, gdal.GA_ReadOnly),
                                 options=opt_string)
        if not dataset:
            raise FileNotFoundError(path)
        return np.array(dataset.ReadAsArray())

    def load_elevation(self, path: str) -> np.array:
        return self.__load_1d_raster(path)

    def load_slope(self, path: str) -> np.array:
        t = self.__load_1d_raster(path)
        t[0, :] = np.zeros_like(t[0, :])
        t[-1, :] = np.zeros_like(t[-1, :])
        t[:, 0] = np.zeros_like(t[:, 0])
        t[:, -1] = np.zeros_like(t[:, -1])
        return t

    def load_optical(self, path_r: str, path_g: str, path_b: str) -> np.array:
        optical_r = self.__load_1d_uint16(path_r)
        optical_g = self.__load_1d_uint16(path_g)
        optical_b = self.__load_1d_uint16(path_b)

        self.parse_meta_with_gdal(path_r) # to save gdal meta for writing
        return np.dstack((optical_r, optical_g, optical_b))

    def load_optical_landsat(self, path_r: str, path_g: str, path_b: str) -> np.array:
        # optical_r = self.__load_1d_raster(path_r)
        # optical_g = self.__load_1d_raster(path_g)
        # optical_b = self.__load_1d_raster(path_b)

        optical_r = self.__load_1d_uint16(path_r)
        optical_g = self.__load_1d_uint16(path_g)
        optical_b = self.__load_1d_uint16(path_b)

        self.parse_meta_with_gdal(path_r) # to save gdal meta for writing
        return np.dstack((optical_r, optical_g, optical_b))

    def load_nir(self, path: str) -> np.array:
        # ToDo check whether this is the best approach to load this channel
        return self.__load_1d_uint16(path)

    def load_nir_landsat(self, path: str) -> np.array:
        # ToDo check whether this is the best approach to load this channel
        # return self.__load_1d_raster(path)
        return self.__load_1d_uint16(path)

    def load_ultrablue(self, path: str) -> np.array:
        # ToDo check whether this is the best approach to load this channel
        return self.__load_1d_uint16(path)

    def load_swir1(self, path: str) -> np.array:
        # ToDo check whether this is the best approach to load this channel
        return self.__load_1d_uint16(path)

    def load_swir1_landsat(self, path: str) -> np.array:
        # ToDo check whether this is the best approach to load this channel
        # return self.__load_1d_raster(path)
        return self.__load_1d_uint16(path)

    def load_swir2(self, path: str) -> np.array:
        # ToDo check whether this is the best approach to load this channel
        return self.__load_1d_uint16(path)

    def load_swir2_landsat(self, path: str) -> np.array:
        # ToDo check whether this is the best approach to load this channel
        # return self.__load_1d_raster(path)
        return self.__load_1d_uint16(path)

    def load_panchromatic(self, path: str) -> np.array:
        return self.__load_1d_raster(path)

    def load_features(self, path: str) -> np.array:
        dataset = gdal.Open(path, gdal.GA_ReadOnly)
        if not dataset:
            raise FileNotFoundError(path)
        signed_data = np.array(dataset.ReadAsArray()).astype(np.int)
        return signed_data - 1

    def load_curve(self, path: str) -> np.array:
        return self.__load_1d_raster(path)

    def load_erosion(self, path: str) -> np.array:
        return self.__load_1d_raster(path)

    def load_roughness(self, path: str) -> np.array:
        return self.__load_1d_raster(path)

    def load_log_roughness(self, path: str) -> np.array:
        log_roughness = self.__load_1d_raster(path)
        log_roughness[log_roughness < 0] = 0
        return log_roughness

    def load_log_flow(self, path: str) -> np.array:
        return self.__load_1d_raster(path)

    def parse_meta_with_gdal(self, path: str):
        """to be used for parsing gdal headers and recreating them in output results
           based on https://www.gdal.org/gdal_tutorial.html
        """

        dataset = gdal.Open(path, gdal.GA_ReadOnly)
        if not dataset:
            raise FileNotFoundError(dataset)

        logging.info("Driver: {}/{}".format(dataset.GetDriver().ShortName,
                                     dataset.GetDriver().LongName))
        self.driver_name = dataset.GetDriver().ShortName

        logging.info("Projection is {}".format(dataset.GetProjection()))
        self.projection = dataset.GetProjection()

        self.geotransform = dataset.GetGeoTransform()
        if self.geotransform:
            logging.info("Origin = ({}, {})".format(self.geotransform[0], self.geotransform[3]))
            logging.info("Pixel Size = ({}, {})".format(self.geotransform[1], self.geotransform[5]))

    def write_image(self, path, image, crop=None):
        if crop is None:
            # image is self.optical_rgb.shape[0] X self.optical_rgb.shape[1] in this case
            driver = gdal.GetDriverByName(self.driver_name)
            if not driver:
                raise Exception("driver not created")
            if image.ndim == 3:
                bands = image.shape[2]
            elif image.ndim == 2:
                bands = 1
            else:
                raise Exception("Bands number incorrect")
            dst_ds = driver.Create(path+".tif", xsize=image.shape[1], ysize=image.shape[0], bands=bands, eType=gdal.GDT_Byte)

            dst_ds.SetGeoTransform(self.geotransform)
            dst_ds.SetProjection(self.projection)
            raster = image.astype(np.uint8)
            if image.ndim == 3:
                for band_ind in range(bands):
                    dst_ds.GetRasterBand(band_ind + 1).WriteArray(raster[:, :, band_ind])
            elif image.ndim == 2:
                dst_ds.GetRasterBand(1).WriteArray(raster)
            dst_ds = None
        else:
            image = image[crop[0]:crop[2], crop[1]:crop[3]]
            plt.imsave(path+".png", image)

    def write_float(self, path, image, crop=None):
        if crop is None:
            # image is self.optical_rgb.shape[0] X self.optical_rgb.shape[1] in this case
            driver = gdal.GetDriverByName(self.driver_name)
            if not driver:
                raise Exception("driver not created")
            if image.ndim == 3:
                bands = image.shape[2]
            elif image.ndim == 2:
                bands = 1
            else:
                raise Exception("Bands number incorrect")
            dst_ds = driver.Create(path+".tif", xsize=image.shape[1], ysize=image.shape[0], bands=bands, eType=gdal.GDT_Float64)

            dst_ds.SetGeoTransform(self.geotransform)
            dst_ds.SetProjection(self.projection)
            raster = image.astype(np.float64)
            if image.ndim == 3:
                for band_ind in range(bands):
                    dst_ds.GetRasterBand(band_ind + 1).WriteArray(raster[:, :, band_ind])
            elif image.ndim == 2:
                dst_ds.GetRasterBand(1).WriteArray(raster)
            dst_ds = None
        else:
            image = image[crop[0]:crop[2], crop[1]:crop[3]]
            plt.imsave(path+".png", image)

    def write_surface(self, path, image, crop=None):
        #todo use gdal dem

        # driver = self.gdal_options['driver']
        # if not driver:
        #     raise Exception("driver not created")
        # if image.ndim == 3:
        #     bands = image.shape[2]
        # elif image.ndim == 2:
        #     bands = 1
        # else:
        #     raise Exception("Bands number incorrect")
        # dst_ds = driver.Create(path, xsize=image.shape[0], ysize=image.shape[1], bands=bands, eType=gdal.GDT_Byte)
        #
        # geotransform = self.gdal_options['geotransform']
        # dst_ds.SetGeoTransform(geotransform)
        # projection = self.gdal_options['projection']
        # dst_ds.SetProjection(projection)
        # dst_ds.GetRasterBand(1).WriteArray((image * 100).astype(np.int))
        #
        # opts = gdal.DEMProcessingOptions(colorFilename='gdal_vis_opts.txt')
        # dst_ds2 = gdal.DEMProcessing(destName="gdal_test.tif", srcDS=dst_ds, processing="color-relief", options=opts)
        # opts2 = gdal.TranslateOptions(format='VRT')
        # dst_ds3 = gdal.Translate(destName="gdal_test.vrt", srcDS=dst_ds2, options=opts2)
        # gdal.BuildVRT()
        # gdal.Translate('heatmaps_3_colours_tmp2.tif', dst_ds)
        # dst_ds_2 = None
        # dst_ds = None

        cmap = plt.get_cmap('jet')
        if crop is None:
            rgba_img_faults = cmap(image)
            rgb_img_faults = np.delete(rgba_img_faults, 3, 2)
            rgb_img_faults=(rgb_img_faults[:, :, :3] * 255).astype(np.uint8)
            self.write_image(path, rgb_img_faults)
        else:
            image = image[crop[0]:crop[2], crop[1]:crop[3]]
        im = plt.imshow(image, cmap=cmap)
        plt.colorbar(im)
        plt.savefig(path+".png")
        plt.close('all')

    def append_additional_features(self, path, features):
        path = Path(path)
        if not path.is_dir():
            logging.warning("additional_features are not specified")
            return features

        utm_coord = UtmCoord(self.geotransform[0], self.geotransform[1], self.geotransform[3], self.geotransform[5])
        lookalike_patches = utm_coord.read_geometry(list(path.glob('Look_Alike_*.kml.utm')))
        nonfault_patches = utm_coord.read_geometry(list(path.glob('Not_Fault_*.kml.utm')))
        lookalike_lines = utm_coord.read_geometry(list(path.glob('Line_Look_Alike_*.kml.utm')))
        fault_lines = utm_coord.read_geometry(list(path.glob('Fault_*.kml.utm')))

        #todo we assign only normal labelling here, as they are patches, not lines and we cn't sample from patches for
        # lookalikes - then the probability of important labelled lines will be low
        for patch in lookalike_patches:
            img = Image.new('L', (features.shape[1], features.shape[0]), 0)
            ImageDraw.Draw(img).polygon(patch, outline=1, fill=1)
            mask = np.array(img)
            features[mask == 1] = FeatureValue.NONFAULT.value

        for patch in nonfault_patches:
            img = Image.new('L', (features.shape[1], features.shape[0]), 0)
            ImageDraw.Draw(img).polygon(patch, outline=1, fill=1)
            mask = np.array(img)
            features[mask == 1] = FeatureValue.NONFAULT.value

        for line in lookalike_lines:
            img = Image.new('L', (features.shape[1], features.shape[0]), 0)
            ImageDraw.Draw(img).line(line, width=2, fill=1)
            mask = np.array(img)
            features[mask == 1] = FeatureValue.FAULT_LOOKALIKE.value

        for line in fault_lines:
            img = Image.new('L', (features.shape[1], features.shape[0]), 0)
            ImageDraw.Draw(img).line(line, width=2, fill=1)
            mask = np.array(img)
            features[mask == 1] = FeatureValue.FAULT.value

        return features

    def get_params(self):
        return {
            'driver_name': self.driver_name,
            'projection': self.projection,
            'geotransform': str(self.geotransform),
        }

    def set_params(self, driver_name, projection, geotransform):
        self.driver_name = driver_name
        self.projection = projection
        self.geotransform = geotransform
