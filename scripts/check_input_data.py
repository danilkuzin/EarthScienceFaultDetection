from osgeo import gdal
import numpy as np
import os

# import rasterio

input_folder = '/Users/olgaisupova/Documents/GitHub/DataForEarthScienceFaultDetection/raw_data/NCAL_Files_sar/'
file_prefix = 'NCAL_'
files = ['B2_UTM.tif', 'B3_UTM.tif', 'B4_UTM.tif', 'B5_UTM.tif',
         'B6_UTM.tif', 'B7_UTM.tif', 'Elev_UTM.tif',
         'Accumulation_LOG_10_UTM.tif', 'TRI_LOG_10_UTM.tif',
         'Mask_UTM.tif', 'HH_LOG_10_UTM.tif', 'HV_LOG_10_UTM.tif']

for file in files:
    full_name = os.path.join(
        input_folder,
        file_prefix + file
    )

    data = np.array(gdal.Open(full_name, gdal.GA_ReadOnly).ReadAsArray())
    # data_rasterio = rasterio.open(full_name).read(1)

    print('-'*50)
    print(f'channel {file}:')
    print('Read with gdal')
    print(f'type {data.dtype}')
    print(f'size {data.shape}')
    print(f'min {np.min(data)}')
    print(f'max {np.max(data)}')

    # print()
    # print('Read with rasterio')
    # print(f'type {data_rasterio.dtype}')
    # print(f'size {data_rasterio.shape}')
    # print(f'min {np.min(data_rasterio)}')
    # print(f'max {np.max(data_rasterio)}')




