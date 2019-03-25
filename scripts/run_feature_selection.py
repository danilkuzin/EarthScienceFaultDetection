import logging
import os

from typing import List, Tuple

import numpy as np
import tensorflow as tf
import pathlib
import h5py
import pandas

from src.DataPreprocessor.data_generator import DataGenerator
from src.DataPreprocessor.data_preprocessor import Mode
from src.LearningKeras.net_architecture import cnn_150x150x5_3class, cnn_150x150x5, \
    cnn_150x150x3, cnn_150x150x1, cnn_150x150x12, cnn_150x150x11, cnn_150x150x4, cnn_150x150x1_3class, \
    cnn_150x150x3_3class, cnn_150x150x10, alexnet, cnn_150x150x5_2class_3convolutions
# from src.LearningKeras.net_architecture import CnnModel150x150x5
import matplotlib.pyplot as plt

# use Pipeline instead
from src.pipeline import global_params

# todo sample validation set at beginning, once
def run(class_probabilities: str, start_channels: List[int], add_channels:List[int], output_path="", epochs=5):
    np.random.seed(1)
    tf.set_random_seed(2)

    with h5py.File(output_path + '/data_fixed.h5', 'r') as hf:
        imgs_valid=hf['imgs_valid'][:]
        lbls_valid = hf['lbls_valid'][:]
        imgs_train = hf['imgs_train'][:]
        lbls_train = hf['lbls_train'][:]

    model = cnn_150x150x3()

    callbacks = [
        tf.keras.callbacks.CSVLogger(filename='log.csv', separator=',', append=False)
    ]

    history = model.fit(x=imgs_train[:, :, :, start_channels],
                        y=lbls_train,
                        epochs=epochs,
                        verbose=2,
                        callbacks=callbacks,
                        validation_data=(imgs_valid[:, :, :, start_channels], lbls_valid),
                        workers=0,
                        use_multiprocessing=False)
    cur_quality = history.history['val_acc']
    quality = cur_quality[-1]
    print(f"initial quality:{quality}")
    tf.keras.backend.clear_session()

    for add_ch in add_channels:
        np.random.seed(1)
        tf.set_random_seed(2)

        init_channels = start_channels.copy()
        init_channels.append(add_ch)
        model = cnn_150x150x4()
        callbacks = [
            tf.keras.callbacks.CSVLogger(filename=f'log_{add_ch}.csv', separator=',', append=False)
        ]
        history = model.fit(x=imgs_train[:, :, :,init_channels],
                            y=lbls_train,
                            epochs=epochs,
                            verbose=2,
                            callbacks=callbacks,
                            validation_data=(imgs_valid[:, :, :, init_channels], lbls_valid),
                            workers=0,
                            use_multiprocessing=False)
        cur_quality = history.history['val_acc']
        quality = cur_quality[-1]
        print(f"add feature:{add_ch}, quality: {quality}")
        tf.keras.backend.clear_session()

def collect_results(path):
    path = pathlib.Path(path)
    res_files = list(path.glob('log_*.csv'))
    val_acc_x = []
    val_acc_y = []
    for res_file in res_files:
        cur_data = pandas.read_csv(res_file)
        val_acc_y.append(cur_data["val_acc"].iloc[-1])
        file_name = os.path.basename(res_file)
        val_acc_x.append(file_name[4])

    val_acc_x = np.array(val_acc_x)
    val_acc_y = np.array(val_acc_y)
    ind = np.argsort(val_acc_x)

    plt.plot(val_acc_x[ind], val_acc_y[ind])
    plt.show()


if __name__ == "__main__":

    # run(class_probabilities="two-class",
    #     start_channels=[0, 1, 2],
    #     add_channels = [6, 7, 8, 9],
    #     output_path="train_on_6_features_01234_no_additional_fixed_val/",
    #     epochs=10)

    collect_results(path="train_on_6_features_01234_no_additional_fixed_val/")
