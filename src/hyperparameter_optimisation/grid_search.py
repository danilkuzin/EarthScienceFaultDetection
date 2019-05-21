# choose learning_rate, batch_size,
# try grid search, random_search, bayes_opt
# but first add early stopping
import itertools

import numpy as np
import tensorflow as tf
import h5py

from src.LearningKeras.net_architecture import cnn_150x150x5

lr_grid = [1e-1, 1e-2, 1e-3, 1e-4]
bs_grid = [5, 10, 15, 20, 25]

epochs = 5

np.random.seed(1)
tf.set_random_seed(2)

with h5py.File('../../scripts/train_on_6_features_01234_no_additional_fixed_val/data_fixed.h5', 'r') as hf:
    imgs_valid=hf['imgs_valid'][:]
    lbls_valid = hf['lbls_valid'][:]
    imgs_train = hf['imgs_train'][:]
    lbls_train = hf['lbls_train'][:]

model = cnn_150x150x5()

callbacks = [
    tf.keras.callbacks.CSVLogger(filename='log.csv', separator=',', append=False)
]

for lr, bs in itertools.product(lr_grid, bs_grid):
    model = cnn_150x150x5()
    callbacks = [
        tf.keras.callbacks.CSVLogger(filename=f'log_lr_{lr}_bs_{bs}.csv', separator=',', append=False),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto',
                                         baseline=None)
    ]
    history = model.fit(x=imgs_train[:, :, :, [0, 1, 2, 3, 4]],
                        y=lbls_train,
                        epochs=epochs,
                        verbose=2,
                        callbacks=callbacks,
                        validation_data=(imgs_valid[:, :, :, [0, 1, 2, 3, 4]], lbls_valid),
                        workers=0,
                        use_multiprocessing=False)
    cur_quality = history.history['val_acc']
    quality = cur_quality[-1]
    print(f"lr: {lr}, bs:{bs}, quality: {quality}")
    tf.keras.backend.clear_session()