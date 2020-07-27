import yaml
import numpy
import seaborn
import matplotlib.pyplot

from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend

saved_image_path = '/Users/danilkuzin/Google Drive Personal/Work/fault_detection/data/results (1)/test_training_segmentation_on_6_torch/prediction_on_6.npz'
mask_prediction = numpy.load(saved_image_path)['prediction']
mask_prediction_max = mask_prediction.max()
mask_prediction_min = mask_prediction.min()
mask_prediction_scaled = (mask_prediction-mask_prediction_min)/(mask_prediction_max-mask_prediction_min)
clip_lower = 0.75
clip_higher = 0.9
mask_prediction_clipped = (mask_prediction_scaled.clip(clip_lower, clip_higher)-clip_lower)/(clip_higher-clip_lower)
seaborn.distplot(mask_prediction_clipped.flatten())
matplotlib.pyplot.show()

geotiff_path = '/Users/danilkuzin/Google Drive Personal/Work/fault_detection/data/results (1)/test_training_segmentation_on_6_torch/prediction_on_6_geo.tiff'


gdal_backend = GdalBackend()
with open(f"/Users/danilkuzin/Documents/danilka/study/github/DataForEarthScienceFaultDetection/preprocessed/6/gdal_params.yaml", 'r') as stream:
    gdal_params = yaml.safe_load(stream)
gdal_backend.set_params(gdal_params['driver_name'], gdal_params['projection'],
                             eval(gdal_params['geotransform']))

gdal_backend.write_surface(geotiff_path, mask_prediction_clipped)