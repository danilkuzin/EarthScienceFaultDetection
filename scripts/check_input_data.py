from osgeo import gdal
import numpy as np
import os

input_folder = '/Users/olgaisupova/Documents/GitHub/DataForEarthScienceFaultDetection/raw_data/NCAL_Files/'
file_prefix = 'NCAL_'
files = ['B2_UTM.tif', 'B3_UTM.tif', 'B4_UTM.tif', 'B5_UTM.tif',
         'B6_UTM.tif', 'B7_UTM.tif', 'Elev_UTM.tif',
         'Log10_FLOW_Scaled_UTM.tif', 'TRI_UTM.tif',
         'Erode_UTM.tif']

for file in files:
    full_name = os.path.join(
        input_folder,
        file_prefix + file
    )

    data = np.array(gdal.Open(full_name, gdal.GA_ReadOnly).ReadAsArray())

    print('-'*50)
    print(f'channel {file}:')
    print(f'type {data.dtype}')
    print(f'size {data.shape}')
    print(f'min {np.min(data)}')
    print(f'max {np.max(data)}')




