import yaml
import numpy
import seaborn
import matplotlib.pyplot as plt

from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.config import data_path

folder = f"{data_path}/results/test_training_segmentation_unet_on_6_torch"
saved_image_path = folder + '/prediction_on_6.npz'
mask_prediction = numpy.load(saved_image_path)['prediction']
mask_prediction_max = mask_prediction.max()
mask_prediction_min = mask_prediction.min()
mask_prediction_scaled = (mask_prediction-mask_prediction_min)/(mask_prediction_max-mask_prediction_min)
clip_lower = 0.6
clip_higher = 1 #0.9
mask_prediction_clipped = (mask_prediction_scaled.clip(clip_lower, clip_higher)-clip_lower)/(clip_higher-clip_lower)
seaborn.distplot(mask_prediction_clipped.flatten())
plt.show()

geotiff_path = folder + '/prediction_on_6_geo.tiff'

gdal_backend = GdalBackend()
with open(f"{data_path}/preprocessed/6/gdal_params.yaml", 'r') as stream:
    gdal_params = yaml.safe_load(stream)
gdal_backend.set_params(gdal_params['driver_name'], gdal_params['projection'],
                             eval(gdal_params['geotransform']))

gdal_backend.write_surface(geotiff_path, mask_prediction_clipped)

plt.imshow(mask_prediction)
plt.show()
