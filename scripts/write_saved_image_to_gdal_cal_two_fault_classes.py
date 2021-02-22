import yaml
import numpy
import seaborn
import matplotlib.pyplot as plt

from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.config import data_path

folder = f"{data_path}/results/unet_on_12_rgb_elev_tri_flow_hazmap_dice_focal_alpha_0.9_semisupervised"
prediction_region = 12
saved_image_path = folder + f'/prediction_on_{prediction_region}.npz'
prediction = numpy.load(saved_image_path)['prediction']
mask_prediction_strike_slip = prediction[1]
mask_prediction_strike_slip_max = mask_prediction_strike_slip.max()
mask_prediction_strike_slip_min = mask_prediction_strike_slip.min()
mask_prediction_strike_slip_scaled = \
    (mask_prediction_strike_slip-mask_prediction_strike_slip_min)/\
    (mask_prediction_strike_slip_max-mask_prediction_strike_slip_min)
clip_lower = 0.5 # 0.3 - hazmap on 6
clip_higher = 1 # 0.7 - hazmap on 6
mask_prediction_strike_slip_clipped = (mask_prediction_strike_slip_scaled.clip(clip_lower, clip_higher)-clip_lower)/(clip_higher-clip_lower)
seaborn.distplot(mask_prediction_strike_slip_clipped.flatten())
plt.title("strike slip")
plt.show()

mask_prediction_thrust = prediction[2]
mask_prediction_thrust_max = mask_prediction_thrust.max()
mask_prediction_thrust_min = mask_prediction_thrust.min()
mask_prediction_thrust_scaled = (mask_prediction_thrust-mask_prediction_thrust_min)/(mask_prediction_thrust_max-mask_prediction_thrust_min)
clip_lower = 0.5
clip_higher = 1 #0.9
mask_prediction_thrust_clipped = (mask_prediction_thrust_scaled.clip(clip_lower, clip_higher)-clip_lower)/(clip_higher-clip_lower)
seaborn.distplot(mask_prediction_thrust_clipped.flatten())
plt.title("thrust")
plt.show()

class_prediction = numpy.argmax(prediction, axis=0)
mask_strike_slip_binary = ((class_prediction == 1) * 255).astype(numpy.uint8)
mask_thrust_binary = ((class_prediction == 2) * 255).astype(numpy.uint8)

geotiff_path = folder + f'/prediction_strike_slip_on_{prediction_region}_geo.tiff'

gdal_backend = GdalBackend()
with open(f"{data_path}/preprocessed/{prediction_region}/gdal_params.yaml", 'r') as stream:
    gdal_params = yaml.safe_load(stream)
gdal_backend.set_params(gdal_params['driver_name'], gdal_params['projection'],
                        eval(gdal_params['geotransform']))

gdal_backend.write_surface(geotiff_path, mask_prediction_strike_slip_clipped)

plt.imshow(mask_prediction_strike_slip_clipped)
plt.title('strike slip')
plt.show()

geotiff_path = folder + f'/prediction_thrust_on_{prediction_region}_geo.tiff'

gdal_backend = GdalBackend()
with open(f"{data_path}/preprocessed/{prediction_region}/gdal_params.yaml", 'r') as stream:
    gdal_params = yaml.safe_load(stream)
gdal_backend.set_params(gdal_params['driver_name'], gdal_params['projection'],
                        eval(gdal_params['geotransform']))

gdal_backend.write_surface(geotiff_path, mask_prediction_thrust_clipped)

plt.imshow(mask_prediction_thrust_clipped)
plt.title('thrust')
plt.show()

geotiff_path = folder + f'/prediction_strike_slip_on_{prediction_region}_geo_binary.tiff'

gdal_backend = GdalBackend()
with open(f"{data_path}/preprocessed/{prediction_region}/gdal_params.yaml", 'r') as stream:
    gdal_params = yaml.safe_load(stream)
gdal_backend.set_params(gdal_params['driver_name'], gdal_params['projection'],
                        eval(gdal_params['geotransform']))

gdal_backend.write_surface(geotiff_path, mask_strike_slip_binary)

plt.imshow(mask_strike_slip_binary)
plt.show()

geotiff_path = folder + f'/prediction_thrust_on_{prediction_region}_geo_binary.tiff'

gdal_backend = GdalBackend()
with open(f"{data_path}/preprocessed/{prediction_region}/gdal_params.yaml", 'r') as stream:
    gdal_params = yaml.safe_load(stream)
gdal_backend.set_params(gdal_params['driver_name'], gdal_params['projection'],
                        eval(gdal_params['geotransform']))

gdal_backend.write_surface(geotiff_path, mask_thrust_binary)

plt.imshow(mask_thrust_binary)
plt.show()


