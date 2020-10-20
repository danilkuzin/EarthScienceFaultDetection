import yaml
import numpy
import seaborn
import matplotlib.pyplot as plt

from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.config import data_path

folder = f"{data_path}/results/unet_on_6_rgb_elev_slope_nir_tri_two_fault_classes_3xlookalike_dice"
prediction_region = 6
saved_image_path = folder + f'/prediction_on_{prediction_region}.npz'
prediction = numpy.load(saved_image_path)['prediction']
mask_prediction_front_range = prediction[1]
mask_prediction_front_range_max = mask_prediction_front_range.max()
mask_prediction_front_range_min = mask_prediction_front_range.min()
mask_prediction_front_range_scaled = (mask_prediction_front_range-mask_prediction_front_range_min)/(mask_prediction_front_range_max-mask_prediction_front_range_min)
clip_lower = 0.2
clip_higher = 0.8 #0.9
mask_prediction_front_range_clipped = (mask_prediction_front_range_scaled.clip(clip_lower, clip_higher)-clip_lower)/(clip_higher-clip_lower)
seaborn.distplot(mask_prediction_front_range_clipped.flatten())
plt.title("front range")
plt.show()

mask_prediction_basin = prediction[2]
mask_prediction_basin_max = mask_prediction_basin.max()
mask_prediction_basin_min = mask_prediction_basin.min()
mask_prediction_basin_scaled = (mask_prediction_basin-mask_prediction_basin_min)/(mask_prediction_basin_max-mask_prediction_basin_min)
clip_lower = 0.6
clip_higher = 1 #0.9
mask_prediction_basin_clipped = (mask_prediction_basin_scaled.clip(clip_lower, clip_higher)-clip_lower)/(clip_higher-clip_lower)
seaborn.distplot(mask_prediction_basin_clipped.flatten())
plt.title("basin")
plt.show()

class_prediction = numpy.argmax(prediction, axis=0)
mask_front_range_binary = ((class_prediction == 1) * 255).astype(numpy.uint8)
mask_basin_binary = ((class_prediction == 2) * 255).astype(numpy.uint8)

geotiff_path = folder + f'/prediction_front_range_on_{prediction_region}_geo.tiff'

gdal_backend = GdalBackend()
with open(f"{data_path}/preprocessed/{prediction_region}/gdal_params.yaml", 'r') as stream:
    gdal_params = yaml.safe_load(stream)
gdal_backend.set_params(gdal_params['driver_name'], gdal_params['projection'],
                        eval(gdal_params['geotransform']))

gdal_backend.write_surface(geotiff_path, mask_prediction_front_range_clipped)

plt.imshow(mask_prediction_front_range_clipped)
plt.title('front range')
plt.show()

geotiff_path = folder + f'/prediction_basin_on_{prediction_region}_geo.tiff'

gdal_backend = GdalBackend()
with open(f"{data_path}/preprocessed/{prediction_region}/gdal_params.yaml", 'r') as stream:
    gdal_params = yaml.safe_load(stream)
gdal_backend.set_params(gdal_params['driver_name'], gdal_params['projection'],
                        eval(gdal_params['geotransform']))

gdal_backend.write_surface(geotiff_path, mask_prediction_basin_clipped)

plt.imshow(mask_prediction_basin_clipped)
plt.title('basin')
plt.show()

geotiff_path = folder + f'/prediction_front_range_on_{prediction_region}_geo_binary.tiff'

gdal_backend = GdalBackend()
with open(f"{data_path}/preprocessed/{prediction_region}/gdal_params.yaml", 'r') as stream:
    gdal_params = yaml.safe_load(stream)
gdal_backend.set_params(gdal_params['driver_name'], gdal_params['projection'],
                        eval(gdal_params['geotransform']))

gdal_backend.write_surface(geotiff_path, mask_front_range_binary)

plt.imshow(mask_front_range_binary)
plt.show()

