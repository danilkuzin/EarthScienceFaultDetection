import sys
import numpy as np
import tensorflow as tf

sys.path.extend(['../../EarthScienceFaultDetection'])

from src.LearningKeras.net_architecture import cnn_150x150x5, cnn_150x150x4
from src.pipeline.training import train_on_preloaded, load_data, datasets_on_single_files, \
    train_on_preloaded_single_files
from src.config import data_path

tf.enable_eager_execution()
np.random.seed(1000)

model = cnn_150x150x5()

# imgs_train, lbls_train, imgs_valid, lbls_valid = load_data(regions=[6], channels=[0, 1, 2, 3, 4], train_ratio=0.80)
# train_on_preloaded(model, imgs_train, lbls_train, imgs_valid, lbls_valid, folder="training_on_6_longer_split_validation", epochs=10)

batch_size = 32

train_dataset, train_dataset_size, valid_dataset, valid_dataset_size = \
    datasets_on_single_files(regions=[6], channels=[0, 1, 2, 3, 4], train_ratio=0.80, batch_size=batch_size)

train_on_preloaded_single_files(model, train_dataset, train_dataset_size, valid_dataset, valid_dataset_size,
                                folder=f"{data_path}/results/training_on_6_only_manual_faults_novel_elevation_normalisation", epochs=10, batch_size=batch_size)
