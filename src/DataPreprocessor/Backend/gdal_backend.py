from tempfile import NamedTemporaryFile

from src.DataPreprocessor.Backend.backend import Backend
import numpy as np
import gdal


class GdalBackend(Backend):
    def load_elevation(self, path: str) -> np.array:
        dataset = gdal.Open(path, gdal.GA_ReadOnly)
        if not dataset:
            raise FileNotFoundError(path)
        return np.array(dataset.ReadAsArray())

    def load_slope(self, path: str) -> np.array:
        dataset = gdal.Open(path, gdal.GA_ReadOnly)
        if not dataset:
            raise FileNotFoundError(path)
        return np.array(dataset.ReadAsArray())

    def load_optical(self, path_r: str, path_g: str, path_b: str) -> np.array:
        opt_string = '-ot Byte -of GTiff -scale 0 65535 0 255'
        # todo check how to remove tmp file
        dataset_r = gdal.Translate(NamedTemporaryFile(delete=False).name, gdal.Open(path_r, gdal.GA_ReadOnly),
                                   options=opt_string)
        optical_r = np.array(dataset_r.ReadAsArray())
        dataset_g = gdal.Translate(NamedTemporaryFile(delete=False).name, gdal.Open(path_g, gdal.GA_ReadOnly),
                                   options=opt_string)
        optical_g = np.array(dataset_g.ReadAsArray())
        dataset_b = gdal.Translate(NamedTemporaryFile(delete=False).name, gdal.Open(path_b, gdal.GA_ReadOnly),
                                   options=opt_string)
        optical_b = np.array(dataset_b.ReadAsArray())

        return np.dstack((optical_r, optical_g, optical_b))

    def load_features(self, path: str) -> np.array:
        dataset = gdal.Open(path, gdal.GA_ReadOnly)
        if not dataset:
            raise FileNotFoundError(path)
        return np.array(dataset.ReadAsArray())


    # to be used for parsing gdal headers and recreating them in output results
    def parse_meta_with_gdal(self):
        # based on https://www.gdal.org/gdal_tutorial.html
        #Opening the File
        dataset = gdal.Open(self.data_dir+self.filename_prefix + '_R.tif', gdal.GA_ReadOnly)
        if not dataset:
            raise GdalFileException()

        scale = '-scale 0 65535 0 255'
        options_list = [
            '-ot Byte',
            '-of GTiff',
            scale
        ]
        options_string = " ".join(options_list)

        dataset = gdal.Translate(self.data_dir+'tmp.tif', dataset, options=options_string)

        #arr = dataset.ReadAsArray()

        # Getting Dataset Information
        print("Driver: {}/{}".format(dataset.GetDriver().ShortName,
                                     dataset.GetDriver().LongName))
        self.gdal_options['driver'] = dataset.GetDriver()

        print("Size is {} x {} x {}".format(dataset.RasterXSize,
                                            dataset.RasterYSize,
                                            dataset.RasterCount))
        self.gdal_options['size'] = [dataset.RasterXSize, dataset.RasterYSize,  dataset.RasterCount]

        print("Projection is {}".format(dataset.GetProjection()))
        self.gdal_options['projection'] = dataset.GetProjection()
        geotransform = dataset.GetGeoTransform()
        if geotransform:
            print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
            print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))
        self.gdal_options['geotransform'] = geotransform

        # Fetching a Raster Band
        band = dataset.GetRasterBand(1)
        print("Band Type={}".format(gdal.GetDataTypeName(band.DataType)))

        min = band.GetMinimum()
        max = band.GetMaximum()
        if not min or not max:
            (min, max) = band.ComputeRasterMinMax(True)
        print("Min={:.3f}, Max={:.3f}".format(min, max))

        if band.GetOverviewCount() > 0:
            print("Band has {} overviews".format(band.GetOverviewCount()))

        if band.GetRasterColorTable():
            print("Band has a color table with {} entries".format(band.GetRasterColorTable().GetCount()))

        #dataset = gdal.Translate(self.data_dir+'tmp.tif', dataset, options=gdal.TranslateOptions(outputType=gdal.GDT_Byte, scaleParams=[0, 65535, 0, 255]))
        #dataset = gdal.Translate(self.data_dir+'tmp.tif', gdal.TranslateOptions(["-of", "GTiff", "-ot", "Byte", "-scale", "0 65535 0 255"]))

        # Reading Raster Data
        scanline = band.ReadRaster(xoff=0, yoff=0,
                                   xsize=band.XSize, ysize=1,
                                   buf_xsize=band.XSize, buf_ysize=1,
                                   buf_type=gdal.GDT_Byte)

        #tuple_of_floats = struct.unpack('f' * band.XSize, scanline)
        tuple_of_floats = struct.unpack('b' * band.XSize, scanline)

        dataset = None

    def write_array(self, backend, image):
        # image is self.optical_rgb.shape[0] X self.optical_rgb.shape[1] in this case
        if backend == Backend.GDAL:
            driver = self.gdal_options['driver']
            dst_ds = driver.Create("out_im", xsize=self.optical_rgb.shape[0], ysize=self.optical_rgb.shape[1],
                                   bands=1, eType=gdal.GDT_Byte)

            geotransform = self.gdal_options['geotransform']
            dst_ds.SetGeoTransform(geotransform)
            projection = self.gdal_options['projection']
            dst_ds.SetProjection(projection)
            raster = image.astype(np.uint8)
            dst_ds.GetRasterBand(1).WriteArray(raster)

            dst_ds = None
        else:
            raise NotImplementedError