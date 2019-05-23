from PIL import Image, ImageDraw
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

from src.DataPreprocessor.DataIOBackend.backend import DataIOBackend
import numpy as np
import gdal
import logging

import matplotlib.pyplot as plt

from src.DataPreprocessor.data_preprocessor import FeatureValue


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
        t = self.__load_1d_raster(path)
        t[0, :] = np.zeros_like(t[0, :])
        t[-1, :] = np.zeros_like(t[-1, :])
        t[:, 0] = np.zeros_like(t[:, 0])
        t[:, -1] = np.zeros_like(t[:, -1])
        return t

    def load_optical(self, path_r: str, path_g: str, path_b: str) -> np.array:
        opt_string = '-ot Byte -of GTiff -scale 0 65535 0 255'
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

    def load_ultrablue(self, path: str) -> np.array:
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

    def load_curve(self, path: str) -> np.array:
        return self.__load_1d_raster(path)

    def load_erosion(self, path: str) -> np.array:
        return self.__load_1d_raster(path)

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

    def write_image(self, path, image, crop=None):
        if crop is None:
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
            dst_ds = driver.Create(path+".tif", xsize=image.shape[1], ysize=image.shape[0], bands=bands, eType=gdal.GDT_Byte)
    
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

        gt = self.gdal_options['geotransform']

        # load lookalike patches
        lookalike_files = list(path.glob('Look_Alike_*.kml.utm'))
        logging.info(f"found lookalike files: {lookalike_files}")
        lookalike_patches = []
        for lookalike_file in lookalike_files:
            with open(str(lookalike_file)) as f:
                content = f.readlines()

            coords = []
            for line in content:
                rr = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line)
                if len(rr) == 3:
                    coords.append(list(map(float, rr)))

            pixel_coords = []

            for coord in coords:
                Xpixel = int((coord[0] - gt[0]) / gt[1])
                Ypixel = int((coord[1] - gt[3]) / gt[5])
                pixel_coords.append((Xpixel, Ypixel))

            lookalike_patches.append(pixel_coords)
        logging.info(f"extracted lookalike patches: {lookalike_patches}")

        # load nonfault patches
        nonfault_files = list(path.glob('Not_Fault_*.kml.utm'))
        logging.info(f"found nonfault files: {nonfault_files}")
        nonfault_patches = []
        for nonfault_file in nonfault_files:
            with open(str(nonfault_file)) as f:
                content = f.readlines()

            coords = []
            for line in content:
                rr = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line)
                if len(rr) == 3:
                    coords.append(list(map(float, rr)))

            pixel_coords = []

            for coord in coords:
                Xpixel = int((coord[0] - gt[0]) / gt[1])
                Ypixel = int((coord[1] - gt[3]) / gt[5])
                pixel_coords.append((Xpixel, Ypixel))

            nonfault_patches.append(pixel_coords)
        logging.info(f"extracted nonfault patches: {nonfault_patches}")

        # load lookalike lines
        lookalike_line_files = list(path.glob('Line_Look_Alike_*.kml.utm'))
        logging.info(f"found lookalike line files: {lookalike_line_files}")
        lookalike_lines = []
        for lookalike_line in lookalike_line_files:
            with open(str(lookalike_line)) as f:
                content = f.readlines()

            coords = []
            for line in content:
                rr = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line)
                if len(rr) == 3:
                    coords.append(list(map(float, rr)))

            pixel_coords = []

            for coord in coords:
                Xpixel = int((coord[0] - gt[0]) / gt[1])
                Ypixel = int((coord[1] - gt[3]) / gt[5])
                pixel_coords.append((Xpixel, Ypixel))

            lookalike_lines.append(pixel_coords)
        logging.info(f"extracted lookalike lines: {lookalike_lines}")

        # load fault lines
        fault_line_files = list(path.glob('Fault_*.kml.utm'))
        logging.info(f"found fault line files: {fault_line_files}")
        fault_lines = []
        for fault_line in fault_line_files:
            with open(str(fault_line)) as f:
                content = f.readlines()

            coords = []
            for line in content:
                rr = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line)
                if len(rr) == 3:
                    coords.append(list(map(float, rr)))

            pixel_coords = []

            for coord in coords:
                Xpixel = int((coord[0] - gt[0]) / gt[1])
                Ypixel = int((coord[1] - gt[3]) / gt[5])
                pixel_coords.append((Xpixel, Ypixel))

            fault_lines.append(pixel_coords)
        logging.info(f"extracted fault lines: {fault_lines}")

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

