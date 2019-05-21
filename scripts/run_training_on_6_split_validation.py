import sys
import numpy as np

sys.path.extend(['../../EarthScienceFaultDetection'])

from src.LearningKeras.net_architecture import cnn_150x150x5
from src.pipeline.training import train_on_preloaded, load_data

np.random.seed(1000)

model = cnn_150x150x5()

imgs_train, lbls_train, imgs_valid, lbls_valid = load_data(regions=[6], channels=[0, 1, 2, 3, 4], train_ratio=0.80)
train_on_preloaded(model, imgs_train, lbls_train, imgs_valid, lbls_valid, folder="training_on_6_longer_split_validation", epochs=10)
