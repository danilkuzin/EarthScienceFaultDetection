import sys
import numpy as np

sys.path.extend(['../../EarthScienceFaultDetection'])

from src.pipeline.training import train_on_preloaded, load_data, datasets_on_single_files, \
    train_on_preloaded_single_files
from src.LearningKeras.net_architecture import cnn_150x150x5
import tensorflow as tf

tf.enable_eager_execution()
np.random.seed(1000)

model = cnn_150x150x5()
# imgs_train, lbls_train, imgs_valid, lbls_valid = load_data(regions=[0, 1, 10], channels=[0, 1, 2, 3, 4],
#                                                            train_ratio=0.80)
# train_on_preloaded(model, imgs_train, lbls_train, imgs_valid, lbls_valid,
#                    folder="training_on_0_1_10", epochs=10)
#
train_dataset, valid_dataset = datasets_on_single_files(regions=[0, 1, 10], channels=[0, 1, 2, 3, 4], train_ratio=0.80)

train_on_preloaded_single_files(model, train_dataset, valid_dataset, folder="training_on_0_1_10", epochs=10)

