import pathlib
import sys
sys.path.extend(['../../EarthScienceFaultDetection'])

from src.DataPreprocessor.data_preprocessor import Mode
from src.LearningKeras.net_architecture import cnn_150x150x5
from src.pipeline import global_params
from src.pipeline.nn_visualisation import NnVisualisation
import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

model = cnn_150x150x5()
model.load_weights('updated_heatmaps_trained_on_6/model.h5')
print(model.summary())
nn_visualisation = NnVisualisation(model=model)

output_path = ("../nn_visualisations/")
pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

data_preprocessor = global_params.data_preprocessor_generator(Mode.TRAIN, 0)
#image = data_preprocessor.get_full_image()[0:5]
#image = data_preprocessor.sample_fault_patch(patch_size=(150, 150))[:, :, 0:5]
for i in range(6):
    image, _ = data_preprocessor.sample_fault_patch(patch_size=(150, 150))
den_rgb, den_elev, den_slope = data_preprocessor.denormalise(image)

plt.imsave(output_path+f"input_image_0_2.png", den_rgb)
plt.imsave(output_path+f"input_image_3.png", den_elev)
plt.imsave(output_path+f"input_image_4.png", den_slope)

image=image[:, :, 0:5]
image = np.expand_dims(image, axis=0)
#nn_visualisation.visualise_intermediate_activations(output_path, image)
#nn_visualisation.visualise_convnet_filters(output_path)
nn_visualisation.visualise_heatmaps_activations(output_path, image)
