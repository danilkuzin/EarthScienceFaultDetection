import logging

from typing import List, Tuple

import numpy as np
import tensorflow as tf
import pathlib
import h5py


from src.DataPreprocessor.data_generator import DataGenerator
from src.DataPreprocessor.data_preprocessor import Mode
from src.LearningKeras.net_architecture import cnn_150x150x5_3class, cnn_150x150x5, \
    cnn_150x150x3, cnn_150x150x1, cnn_150x150x12, cnn_150x150x11, cnn_150x150x4, cnn_150x150x1_3class, \
    cnn_150x150x3_3class, cnn_150x150x10, alexnet, cnn_150x150x5_2class_3convolutions
# from src.LearningKeras.net_architecture import CnnModel150x150x5

# use Pipeline instead
from src.pipeline import global_params

def create_datasets(train_datasets: List[int], validation_datasets: List[int], class_probabilities: str,
        patch_size: Tuple[int, int], channels: List[int], output_path="", valid_size=1000, train_size=1000):
    train_preprocessors = [global_params.data_preprocessor_generators[ind](Mode.TRAIN) for ind in train_datasets]
    train_data_generator = DataGenerator(preprocessors=train_preprocessors)
    valid_preprocessors = [global_params.data_preprocessor_generators[ind](Mode.TRAIN) for ind in validation_datasets]
    valid_data_generator = DataGenerator(preprocessors=valid_preprocessors)

    if class_probabilities == "equal":
        class_probabilities_int = np.array([1. / 3, 1. / 3, 1. / 3])
        train_joint_generator = train_data_generator.generator_3class(batch_size=train_size,
                                                                      class_probabilities=class_probabilities_int,
                                                                      patch_size=patch_size,
                                                                      channels=np.array(channels))
        valid_joint_generator = valid_data_generator.generator_3class(batch_size=valid_size,
                                                                      class_probabilities=class_probabilities_int,
                                                                      patch_size=patch_size,
                                                                      channels=np.array(channels))

    elif class_probabilities == "two-class":
        class_probabilities_int = np.array([0.5, 0.25, 0.25])
        train_joint_generator = train_data_generator.generator_2class_lookalikes_with_nonfaults(batch_size=train_size,
                                                                                                class_probabilities=class_probabilities_int,
                                                                                                patch_size=patch_size,
                                                                                                channels=np.array(
                                                                                                    channels))
        valid_joint_generator = valid_data_generator.generator_2class_lookalikes_with_nonfaults(batch_size=valid_size,
                                                                                                class_probabilities=class_probabilities_int,
                                                                                                patch_size=patch_size,
                                                                                                channels=np.array(
                                                                                                    channels))

    else:
        raise Exception('Not implemented')

    imgs_valid, lbls_valid = next(valid_joint_generator)
    imgs_train, lbls_train = next(train_joint_generator)

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path + '/data_fixed.h5', 'w') as hf:
            hf.create_dataset("imgs_valid", data=imgs_valid)
            hf.create_dataset("lbls_valid", data=lbls_valid)
            hf.create_dataset("imgs_train", data=imgs_train)
            hf.create_dataset("lbls_train", data=lbls_train)


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

if __name__ == "__main__":
    # create_datasets(train_datasets=[6],
    #     validation_datasets=[7],
    #     class_probabilities="two-class",
    #     patch_size=(150, 150),
    #     channels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #     output_path="train_on_6_features_01234_no_additional_fixed_val/",
    #                 valid_size=250,
    #                 train_size=1000)

    run(class_probabilities="two-class",
        start_channels=[0, 1, 2],
        add_channels = [6, 7, 8, 9],
        output_path="train_on_6_features_01234_no_additional_fixed_val/",
        epochs=10)
