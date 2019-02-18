from tempfile import NamedTemporaryFile

from src.DataPreprocessor.DataIOBackend.backend import DataIOBackend
import numpy as np
import gdal
import logging

import matplotlib.pyplot as plt


class GdalBackend(DataIOBackend):
    def __init__(self):
        self.gdal_options = dict()

    def __load_1d_raster(self, path: str) -> np.array:
        dataset = gdal.Open(path, gdal.GA_ReadOnly)
        if not dataset:
            raise FileNotFoundError(path)
        return np.array(dataset.ReadAsArray())

    def load_elevation(self, path: str) -> np.array:
        return self.__load_1d_raster(path)

    def load_slope(self, path: str) -> np.array:
        return self.__load_1d_raster(path)

    def load_optical(self, path_r: str, path_g: str, path_b: str) -> np.array:
        opt_string = '-ot Byte -of GTiff -scale 0 65535 0 255'
        # todo check how to remove tmp file and replace with ''
        dataset_r = gdal.Translate(NamedTemporaryFile(delete=False).name, gdal.Open(path_r, gdal.GA_ReadOnly),
                                   options=opt_string)
        optical_r = np.array(dataset_r.ReadAsArray())
        dataset_g = gdal.Translate(NamedTemporaryFile(delete=False).name, gdal.Open(path_g, gdal.GA_ReadOnly),
                                   options=opt_string)
        optical_g = np.array(dataset_g.ReadAsArray())
        dataset_b = gdal.Translate(NamedTemporaryFile(delete=False).name, gdal.Open(path_b, gdal.GA_ReadOnly),
                                   options=opt_string)
        optical_b = np.array(dataset_b.ReadAsArray())

        self.parse_meta_with_gdal(path_r) # to save gdal meta for writing
        return np.dstack((optical_r, optical_g, optical_b))

    def load_nir(self, path: str) -> np.array:
        return self.__load_1d_raster(path)

    def load_ir(self, path: str) -> np.array:
        return self.__load_1d_raster(path)

    def load_swir1(self, path: str) -> np.array:
        return self.__load_1d_raster(path)

    def load_swir2(self, path: str) -> np.array:
        return self.__load_1d_raster(path)

    def load_panchromatic(self, path: str) -> np.array:
        return self.__load_1d_raster(path)

    def load_features(self, path: str) -> np.array:
        dataset = gdal.Open(path, gdal.GA_ReadOnly)
        if not dataset:
            raise FileNotFoundError(path)
        signed_data = np.array(dataset.ReadAsArray()).astype(np.int)
        return signed_data - 1

    def parse_meta_with_gdal(self, path: str):
        """to be used for parsing gdal headers and recreating them in output results
           based on https://www.gdal.org/gdal_tutorial.html
        """
        #Opening the File
        dataset = gdal.Open(path, gdal.GA_ReadOnly)
        if not dataset:
            raise FileNotFoundError(dataset)

        scale = '-scale 0 65535 0 255'
        options_list = [
            '-ot Byte',
            '-of GTiff',
            scale
        ]
        options_string = " ".join(options_list)

        dataset = gdal.Translate(NamedTemporaryFile(delete=False).name, dataset, options=options_string)

        # Getting Dataset Information
        logging.info("Driver: {}/{}".format(dataset.GetDriver().ShortName,
                                     dataset.GetDriver().LongName))
        self.gdal_options['driver'] = dataset.GetDriver()

        logging.info("Size is {} x {} x {}".format(dataset.RasterXSize,
                                            dataset.RasterYSize,
                                            dataset.RasterCount))
        self.gdal_options['size'] = [dataset.RasterXSize, dataset.RasterYSize,  dataset.RasterCount]

        logging.info("Projection is {}".format(dataset.GetProjection()))
        self.gdal_options['projection'] = dataset.GetProjection()
        geotransform = dataset.GetGeoTransform()
        if geotransform:
            logging.info("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
            logging.info("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))
        self.gdal_options['geotransform'] = geotransform

        # Fetching a Raster Band
        #band = dataset.GetRasterBand(1)
        #print("Band Type={}".format(gdal.GetDataTypeName(band.DataType)))

        #min = band.GetMinimum()
        #max = band.GetMaximum()
        #if not min or not max:
        #    (min, max) = band.ComputeRasterMinMax(True)
        #print("Min={:.3f}, Max={:.3f}".format(min, max))

        #if band.GetOverviewCount() > 0:
        #    print("Band has {} overviews".format(band.GetOverviewCount()))

        #if band.GetRasterColorTable():
        #    print("Band has a color table with {} entries".format(band.GetRasterColorTable().GetCount()))

        #dataset = gdal.Translate(self.data_dir+'tmp.tif', dataset, options=gdal.TranslateOptions(outputType=gdal.GDT_Byte, scaleParams=[0, 65535, 0, 255]))
        #dataset = gdal.Translate(self.data_dir+'tmp.tif', gdal.TranslateOptions(["-of", "GTiff", "-ot", "Byte", "-scale", "0 65535 0 255"]))

        # Reading Raster Data
        #scanline = band.ReadRaster(xoff=0, yoff=0,
        #                           xsize=band.XSize, ysize=1,
        #                           buf_xsize=band.XSize, buf_ysize=1,
        #                           buf_type=gdal.GDT_Byte)

        #tuple_of_floats = struct.unpack('f' * band.XSize, scanline)
        #tuple_of_floats = struct.unpack('b' * band.XSize, scanline)

        dataset = None

    def write_image(self, path, image):
        # image is self.optical_rgb.shape[0] X self.optical_rgb.shape[1] in this case
        driver = self.gdal_options['driver']
        if not driver:
            raise Exception("driver not created")
        if image.ndim == 3:
            bands = image.shape[2]
        elif image.ndim == 2:
            bands = 1
        else:
            raise Exception("Bands number incorrect")
        dst_ds = driver.Create(path, xsize=image.shape[0], ysize=image.shape[1], bands=bands, eType=gdal.GDT_Byte)

        geotransform = self.gdal_options['geotransform']
        dst_ds.SetGeoTransform(geotransform)
        projection = self.gdal_options['projection']
        dst_ds.SetProjection(projection)
        raster = image.astype(np.uint8)
        if image.ndim == 3:
            for band_ind in range(bands):
                dst_ds.GetRasterBand(band_ind + 1).WriteArray(raster[:, :, band_ind])
        elif image.ndim == 2:
            dst_ds.GetRasterBand(1).WriteArray(raster)
        dst_ds = None

    def write_surface(self, path, image):
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
        # dst_ds.GetRasterBand(1).WriteArray(image)
        #
        # opts = gdal.DEMProcessingOptions(colorFilename='gdal_vis_opts.txt')
        # dst_ds2 = gdal.DEMProcessing(destName="gdal_test.tif", srcDS=dst_ds, processing="color-relief", options=opts)
        # opts2 = gdal.TranslateOptions(format='VRT')
        # dst_ds3 = gdal.Translate(destName="gdal_test.vrt", srcDS=dst_ds2, options=opts2)
        # gdal.BuildVRT()
        # # gdal.Translate('heatmaps_3_colours_tmp2.tif', dst_ds)
        # # dst_ds_2 = None
        # # dst_ds = None
        cmap = plt.get_cmap('jet')
        rgba_img_faults = cmap(image)
        rgb_img_faults = np.delete(rgba_img_faults, 3, 2)
        rgb_img_faults=(rgb_img_faults[:, :, :3] * 255).astype(np.uint8)
        self.write_image(path, rgb_img_faults)

