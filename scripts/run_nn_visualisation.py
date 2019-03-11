from src.DataPreprocessor.data_preprocessor import Mode
from src.LearningKeras.net_architecture import cnn_150x150x5
from src.pipeline import global_params
from src.pipeline.nn_visualisation import NnVisualisation
import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

np.random.seed(1)
tf.set_random_seed(1)

model = cnn_150x150x5()
model.load_weights('new_labels_all_nonfaults/trained_models_01/model_0.h5')
print(model.summary())
nn_visualisation = NnVisualisation(model=model)

data_preprocessor = global_params.data_preprocessor_generators[0](Mode.TRAIN)
#image = data_preprocessor.get_full_image()[0:5]
#image = data_preprocessor.sample_fault_patch(patch_size=(150, 150))[:, :, 0:5]
image = data_preprocessor.sample_nonfault_patch(patch_size=(150, 150))[:, :, 0:5]
f, ax = plt.subplots(1, 3)
ax[0].imshow(image[:,:, 0:3])
ax[1].imshow(image[:,:, 3])
ax[2].imshow(image[:,:,4])
plt.show()

image = np.expand_dims(image, axis=0)
nn_visualisation.visualise_intermediate_activations(image)
nn_visualisation.visualise_convnet_filters()
nn_visualisation.visualise_heatmaps_activations(image)
