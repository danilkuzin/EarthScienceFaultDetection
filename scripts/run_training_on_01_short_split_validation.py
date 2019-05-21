import sys
import numpy as np

sys.path.extend(['../../EarthScienceFaultDetection'])

from src.LearningKeras.net_architecture import cnn_150x150x5
from src.pipeline.training import load_data, train_on_preloaded

np.random.seed(1000)
model = cnn_150x150x5()

imgs_train, lbls_train, imgs_valid, lbls_valid = load_data(regions=[0, 1], channels=[0, 1, 2, 3, 4], train_ratio=0.50)
train_on_preloaded(model, imgs_train, lbls_train, imgs_valid, lbls_valid, folder="training_on_01_short_split_validation", epochs=5)
