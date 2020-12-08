import gdal
import numpy as np
import os

input_folder = '/Users/danilkuzin/Downloads/CCAL_Files/'
file_prefix = 'CCAL_'
files = ['B2_UTM.tif', 'B3_UTM.tif', 'B4_UTM.tif', 'B5_UTM.tif',
         'B6_UTM.tif', 'B7_UTM.tif', 'Elev_UTM.tif',
         'Flow_Log_Scale.tif', 'TRI_Log_Scale.tif']

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




