import geopandas
import fiona
from osgeo import gdal
import numpy as np
import rasterio
import rasterio.warp
import shapely.geometry

from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.DataPreprocessor.DataIOBackend.utm_coord import UtmCoord

data_folder = '/Users/danilkuzin/Documents/danilka/study/github/DataForEarthScienceFaultDetection/raw_data/CCAL_Files/'
kml_file_name = 'HAZMAP.kml'
geotif_file_name = 'CCAL_B2_UTM.tif'

fiona.drvsupport.supported_drivers['libkml'] = 'rw'
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
# data = fiona.open(data_folder + file_name)

data = geopandas.read_file(data_folder + kml_file_name)

# data_io_backend = GdalBackend()
# data_io_backend.parse_meta_with_gdal(data_folder + geotif_file_name)
#
# dataset = gdal.Open(data_folder + geotif_file_name, gdal.GA_ReadOnly)
# np_image = np.array(dataset.ReadAsArray())
#
# utm_coord = UtmCoord(data_io_backend.geotransform[0],
#                      data_io_backend.geotransform[1],
#                      data_io_backend.geotransform[3],
#                      data_io_backend.geotransform[5])

dataset = rasterio.open(data_folder + geotif_file_name)

latlon_bounds = rasterio.warp.transform_bounds(
    dataset.crs, "EPSG:4326", *dataset.bounds)

filtered_data = data[data.intersects(shapely.geometry.box(*latlon_bounds))]

filtered_data = filtered_data.to_crs(dataset.crs)

# shapely.wkt.loads(filtered_data)

# filtered_data = filtered_data[filtered_data['disp_slip_'] == 'strike slip']

# list(filtered_data.iloc[0]['geometry'].coords)

# for line in filtered_data.loc[394]['geometry']:
#     print(list(line.coords))

# filtered_data.loc[357]['geometry'].type












