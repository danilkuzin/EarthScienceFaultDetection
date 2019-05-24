import pathlib
from typing import List, Tuple

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE

from src.DataPreprocessor.data_generator import DataGenerator
from src.DataPreprocessor.data_preprocessor import Mode
from src.LearningKeras.net_architecture import cnn_150x150x5_3class, cnn_150x150x5, \
    cnn_150x150x3, cnn_150x150x1, cnn_150x150x12, cnn_150x150x11, cnn_150x150x4, cnn_150x150x1_3class, \
    cnn_150x150x3_3class, cnn_150x150x10
from src.pipeline import global_params

def get_class_probabilities_int(class_probabilities):
    class_probabilities_int = None

    if class_probabilities not in ["equal", "two-class"]:
        raise ValueError()

    if class_probabilities == "equal":
        class_probabilities_int = np.array([1. / 3, 1. / 3, 1. / 3])
    elif class_probabilities == "two-class":
        class_probabilities_int = np.array([0.5, 0.25, 0.25])
    return class_probabilities_int


def get_train_joint_generator(class_probabilities, class_probabilities_int, train_data_generator, batch_size,
                              patch_size, channels):
    if class_probabilities == "equal":
        train_joint_generator = train_data_generator.generator_3class(batch_size=batch_size,
                                                                      class_probabilities=class_probabilities_int,
                                                                      patch_size=patch_size,
                                                                      channels=np.array(channels))
    elif class_probabilities == "two-class":
        train_joint_generator = train_data_generator.generator_2class_lookalikes_with_nonfaults(batch_size=batch_size,
                                                                                                class_probabilities=class_probabilities_int,
                                                                                                patch_size=patch_size,
                                                                                                channels=np.array(
                                                                                                    channels))
    else:
        raise ValueError()

    return train_joint_generator


def get_valid_joint_generator(class_probabilities, class_probabilities_int, valid_data_generator, valid_size,
                              patch_size, channels):
    if class_probabilities == "equal":
        valid_joint_generator = valid_data_generator.generator_3class(batch_size=valid_size,
                                                                      class_probabilities=class_probabilities_int,
                                                                      patch_size=patch_size,
                                                                      channels=np.array(channels))
    elif class_probabilities == "two-class":
        valid_joint_generator = valid_data_generator.generator_2class_lookalikes_with_nonfaults(batch_size=valid_size,
                                                                                                class_probabilities=class_probabilities_int,
                                                                                                patch_size=patch_size,
                                                                                                channels=np.array(
                                                                                                    channels))
    else:
        raise ValueError()

    return valid_joint_generator


def get_model(class_probabilities, channels):
    if class_probabilities == "equal":
        if len(channels) == 5:
            model = cnn_150x150x5_3class()
        elif len(channels) == 3:
            model = cnn_150x150x3_3class()
        elif len(channels) == 1:
            model = cnn_150x150x1_3class()
        else:
            raise Exception()

    elif class_probabilities == "two-class":
        if len(channels) == 12:
            model = cnn_150x150x12()
        elif len(channels) == 11:
            model = cnn_150x150x11()
        elif len(channels) == 10:
            model = cnn_150x150x10()
        elif len(channels) == 5:
            model = cnn_150x150x5()
        elif len(channels) == 4:
            model = cnn_150x150x4()
        elif len(channels) == 3:
            model = cnn_150x150x3()
        elif len(channels) == 1:
            model = cnn_150x150x1()
        else:
            raise Exception()

    else:
        raise Exception('Not implemented')
    return model


# todo sample validation set at beginning, once
def train(train_datasets: List[int], test_datasets: List[int], validation_datasets: List[int], class_probabilities: str,
          batch_size: int, patch_size: Tuple[int, int], channels: List[int], output_path="", steps_per_epoch=50,
          epochs=5, valid_size=1000):
    np.random.seed(1)
    tf.set_random_seed(2)

    train_preprocessors = [global_params.data_preprocessor_generators[ind](Mode.TRAIN) for ind in train_datasets]
    train_data_generator = DataGenerator(preprocessors=train_preprocessors)
    # test_preprocessors = [global_params.data_preprocessor_generators[ind](Mode.TRAIN) for ind in test_datasets]
    # test_data_generator = DataGenerator(preprocessors=test_preprocessors)
    valid_preprocessors = [global_params.data_preprocessor_generators[ind](Mode.TRAIN) for ind in validation_datasets]
    valid_data_generator = DataGenerator(preprocessors=valid_preprocessors)

    class_probabilities_int = get_class_probabilities_int(class_probabilities)
    train_joint_generator = get_train_joint_generator(class_probabilities, class_probabilities_int,
                                                      train_data_generator, batch_size, patch_size, channels)
    valid_joint_generator = get_valid_joint_generator(class_probabilities, class_probabilities_int,
                                                      valid_data_generator, valid_size, patch_size, channels)

    model = get_model(class_probabilities, channels)

    #todo test earlystopping
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=32, write_graph=True,
                                       write_grads=True, write_images=True),
        tf.keras.callbacks.CSVLogger(filename='log.csv', separator=',', append=False),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto',
                                         baseline=None)
    ]

    imgs_valid, lbls_valid, _ = next(valid_joint_generator)
    valid_joint_generator = None  # release resources #todo replace with "with"
    valid_preprocessors = None  # release resources

    # todo train_joint_generator may now return 3 arrays instead of two, which is incorrect for fit_generator
    history = model.fit_generator(generator=train_joint_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  verbose=2,
                                  callbacks=callbacks,
                                  validation_data=(imgs_valid, lbls_valid),
                                  workers=0,
                                  use_multiprocessing=False)

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_weights(output_path + '/model_{}.h5'.format(''.join(str(i) for i in train_datasets)))

    # switched to tensorboard + csv
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.savefig("Model accuracy_{}.png".format(hist_ind))
    # plt.close()
    #
    # # Plot training & validation loss values
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.savefig("Model loss_{}.png".format(hist_ind))
    # plt.close()

def load_data(regions, channels, train_ratio):
    imgs_raw = []
    lbls_raw = []
    for reg_id in regions:
        with h5py.File(f'../train_data/regions_{reg_id}/data.h5', 'r') as hf:
            imgs_raw.append(hf['imgs'][:])
            lbls_raw.append(hf['lbls'][:])

    imgs = np.concatenate(imgs_raw, axis=0)
    lbls = np.concatenate(lbls_raw, axis=0)

    permuted_ind = np.random.permutation(imgs.shape[0])
    imgs = imgs[permuted_ind]
    lbls = lbls[permuted_ind]

    imgs = imgs[:, :, :, channels]

    train_len = int(imgs.shape[0] * train_ratio)
    imgs_train = imgs[:train_len].copy()
    lbls_train = lbls[:train_len].copy()
    imgs_valid = imgs[train_len:].copy()
    lbls_valid = lbls[train_len:].copy()

    return imgs_train, lbls_train, imgs_valid, lbls_valid

def train_on_preloaded(model, imgs_train, lbls_train, imgs_valid, lbls_valid, folder, epochs):
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

    callbacks = [
        # tf.keras.callbacks.TensorBoard(log_dir='{folder}/logs', histogram_freq=1, batch_size=32, write_graph=True,
        #                                write_grads=True, write_images=True),
        tf.keras.callbacks.CSVLogger(filename='{folder}/log.csv', separator=',', append=False),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto',
                                         baseline=None)
    ]

    history = model.fit(x=imgs_train,
                        y=lbls_train,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=(imgs_valid, lbls_valid),
                        workers=0,
                        use_multiprocessing=False)

    model.save_weights(folder + '/model.h5')

def train_on_preloaded_single_files(model, train_dataset, train_dataset_size, valid_dataset, valid_dataset_size,
                                    folder, epochs, batch_size):
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=f'{folder}/logs', histogram_freq=1, batch_size=32, write_graph=True,
                                       write_grads=True, write_images=True),
        tf.keras.callbacks.CSVLogger(filename=f'{folder}/log.csv', separator=',', append=False),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto',
                                         baseline=None)
    ]

    history = model.fit(train_dataset,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=valid_dataset,
                        workers=0,
                        use_multiprocessing=False,
                        steps_per_epoch=int(np.ceil(train_dataset_size / batch_size)),
                        validation_steps=int(np.ceil(valid_dataset_size / batch_size)))

    model.save_weights(folder + '/model.h5')

#todo potentially use tfrecord insttead of h5 to maybe store all in one file and remove the py_func from here
def datasets_on_single_files(regions, channels, train_ratio, batch_size):
    BATCH_SIZE = batch_size

    def parse_file(f):
        with h5py.File(f, 'r') as hf:
            img = hf['img'][:].astype(np.float32)[:, :, channels]
            lbl = hf['lbl'][:].astype(np.int32)
            coord = hf['coord'][:].astype(np.int32)
            return img, lbl

    def parse_file_tf(filename):
        return tf.py_func(parse_file, [filename], [tf.float32, tf.int32])

    train_datasets = []
    valid_datasets = []

    train_dataset_size = 0
    valid_dataset_size = 0

    for reg_id in regions:
        reg_path = pathlib.Path(f'../../DataForEarthScienceFaultDetection/train_data/regions_{reg_id}_single_files/')
        all_image_paths = np.array([str(path) for path in list(reg_path.glob('*'))])

        image_count = len(all_image_paths)
        permuted_ind = np.random.permutation(image_count)
        permuted_paths = all_image_paths[permuted_ind]
        train_len = int(image_count * train_ratio)

        train_path_ds = tf.data.Dataset.from_tensor_slices(permuted_paths[:train_len])
        train_image_label_coord_ds = train_path_ds.map(parse_file_tf, num_parallel_calls=AUTOTUNE)
        train_ds = train_image_label_coord_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
        train_ds = train_ds.batch(BATCH_SIZE)
        train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
        train_datasets.append(train_ds)
        train_dataset_size += train_len

        valid_path_ds = tf.data.Dataset.from_tensor_slices(permuted_paths[train_len:])
        valid_image_label_coord_ds = valid_path_ds.map(parse_file_tf, num_parallel_calls=AUTOTUNE)
        valid_ds = valid_image_label_coord_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
        valid_ds = valid_ds.batch(BATCH_SIZE)
        valid_ds = valid_ds.prefetch(buffer_size=AUTOTUNE)
        valid_datasets.append(valid_ds)
        valid_dataset_size += image_count - train_len


    train_dataset = tf.data.experimental.sample_from_datasets(train_datasets)
    valid_dataset = tf.data.experimental.sample_from_datasets(valid_datasets)

    return train_dataset, train_dataset_size, valid_dataset, valid_dataset_size