import logging

from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.DataPreprocessor.PatchesOutputBackend.h5_backend import H5Backend
from src.DataPreprocessor.data_preprocessor import DataPreprocessor, Mode
from src.LearningKeras.net_architecture import cnn_150x150x5, cnn_150x150x5_3class, cnn_150x150x3_3class, \
    cnn_150x150x1_3class
from src.LearningKeras.train import KerasTrainer

import numpy as np
import tensorflow as tf

def process_1_lopukangri():
    np.random.seed(1)
    tf.set_random_seed(2)

    logging.basicConfig(level=logging.WARN)
    dataiobackend = GdalBackend()
    outputbackend = H5Backend()
    loader = DataPreprocessor(data_dir="../data/Region 1 - Lopukangri/",
                              backend=dataiobackend,
                              filename_prefix="tibet",
                              mode=Mode.TRAIN,
                              seed=1)
    #loader.prepare_datasets(output=DataOutput.TFRECORD)
    #loader.prepare_all_patches(backend=outputbackend)
    loader.normalise()

    model_generator = lambda: cnn_150x150x5_3class()
    ensemble_size = 5
    batch_size = 5

    trainer = KerasTrainer(model_generator=model_generator,
                           ensemble_size=ensemble_size,
                           data_preprocessor=loader,
                           batch_size=batch_size)

    train_generator = loader.train_generator(batch_size=batch_size,
                                             class_probabilities=np.array([1./3, 1./3, 1./3]),
                                             patch_size=(150, 150),
                                             channels=np.array([0, 1, 2, 3, 4]))
    trainer.train(steps_per_epoch=50, epochs=5, train_generator=train_generator)
    trainer.apply_for_all_patches()


def process_1_lopukangri_rgb_only():
    np.random.seed(1)
    tf.set_random_seed(2)

    logging.basicConfig(level=logging.WARN)
    dataiobackend = GdalBackend()
    outputbackend = H5Backend()
    loader = DataPreprocessor(data_dir="../data/Region 1 - Lopukangri/",
                              backend=dataiobackend,
                              filename_prefix="tibet",
                              mode=Mode.TRAIN,
                              seed=1)
    #loader.prepare_datasets(output=DataOutput.TFRECORD)
    #loader.prepare_all_patches(backend=outputbackend)
    loader.normalise()

    model_generator = lambda: cnn_150x150x3_3class()
    ensemble_size = 5
    batch_size = 5

    trainer = KerasTrainer(model_generator=model_generator,
                           ensemble_size=ensemble_size,
                           data_preprocessor=loader,
                           batch_size=batch_size)

    train_generator = loader.train_generator(batch_size=batch_size,
                                             class_probabilities=[0.33, 0.33, 0.33],
                                             patch_size=(150, 150),
                                             channels=[0, 1, 2])
    trainer.train(steps_per_epoch=50, epochs=5, train_generator=train_generator)
    trainer.apply_for_all_patches()

def process_1_lopukangri_slope_only():
    np.random.seed(1)
    tf.set_random_seed(2)

    logging.basicConfig(level=logging.WARN)
    dataiobackend = GdalBackend()
    outputbackend = H5Backend()
    loader = DataPreprocessor(data_dir="../data/Region 1 - Lopukangri/",
                              backend=dataiobackend,
                              filename_prefix="tibet",
                              mode=Mode.TRAIN,
                              seed=1)
    #loader.prepare_datasets(output=DataOutput.TFRECORD)
    #loader.prepare_all_patches(backend=outputbackend)
    loader.normalise()

    model_generator = lambda: cnn_150x150x1_3class()
    ensemble_size = 5
    batch_size = 5

    trainer = KerasTrainer(model_generator=model_generator,
                           ensemble_size=ensemble_size,
                           data_preprocessor=loader,
                           batch_size=batch_size)

    train_generator = loader.train_generator(batch_size=batch_size,
                                             class_probabilities=[0.33, 0.33, 0.33],
                                             patch_size=(150, 150),
                                             channels=[4])
    trainer.train(steps_per_epoch=50, epochs=5, train_generator=train_generator)
    trainer.apply_for_all_patches()

def process_1_lopukangri_elevation_only():
    np.random.seed(1)
    tf.set_random_seed(2)

    logging.basicConfig(level=logging.WARN)
    dataiobackend = GdalBackend()
    outputbackend = H5Backend()
    loader = DataPreprocessor(data_dir="../data/Region 1 - Lopukangri/",
                              backend=dataiobackend,
                              filename_prefix="tibet",
                              mode=Mode.TRAIN,
                              seed=1)
    #loader.prepare_datasets(output=DataOutput.TFRECORD)
    #loader.prepare_all_patches(backend=outputbackend)
    loader.normalise()

    model_generator = lambda: cnn_150x150x1_3class()
    ensemble_size = 5
    batch_size = 5

    trainer = KerasTrainer(model_generator=model_generator,
                           ensemble_size=ensemble_size,
                           data_preprocessor=loader,
                           batch_size=batch_size)

    train_generator = loader.train_generator(batch_size=batch_size,
                                                            class_probabilities=[0.33, 0.33, 0.33],
                                                            patch_size=(150, 150),
                                                            channels=[3])
    trainer.train(steps_per_epoch=50, epochs=5, train_generator=train_generator)
    trainer.apply_for_all_patches()

process_1_lopukangri()

