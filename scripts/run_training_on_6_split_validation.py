import pathlib

import h5py
import numpy as np
import tensorflow as tf

import sys
sys.path.extend(['../../EarthScienceFaultDetection'])

from src.LearningKeras.net_architecture import cnn_150x150x3, cnn_150x150x5

with h5py.File('../train_data/regions_6/data.h5', 'r') as hf:
    imgs = hf['imgs'][:]
    lbls = hf['lbls'][:]

np.random.seed(1000)

permuted_ind = np.random.permutation(imgs.shape[0])
imgs = imgs[permuted_ind]
lbls = lbls[permuted_ind]

imgs = imgs[:, :, :, 0:5]

train_ratio = 0.80

train_len = int(imgs.shape[0] * train_ratio)
imgs_train = imgs[:train_len].copy()
lbls_train = lbls[:train_len].copy()
imgs_valid = imgs[train_len:].copy()
lbls_valid = lbls[train_len:].copy()
imgs = None
lbls = None


model = cnn_150x150x5()

callbacks = [
    tf.keras.callbacks.CSVLogger(filename='training_on_6_split_validation/log.csv', separator=',', append=False)
]

output_path = "training_on_6_split_validation"
pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

history = model.fit(x=imgs_train,
                    y=lbls_train,
                    epochs=10,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=(imgs_valid, lbls_valid),
                    workers=0,
                    use_multiprocessing=False)

model.save_weights(output_path + '/model.h5')