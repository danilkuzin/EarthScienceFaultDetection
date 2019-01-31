import logging

from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.DataPreprocessor.PatchesOutputBackend.h5_backend import H5Backend
from src.DataPreprocessor.data_preprocessor import DataPreprocessor, Mode, DataOutput
from src.LearningKeras.net_architecture import cnn_150x150x5
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

    model_generator = lambda: cnn_150x150x5()
    ensemble_size = 5
    batch_size = 5

    trainer = KerasTrainer(model_generator=model_generator,
                           ensemble_size=ensemble_size,
                           data_preprocessor=loader,
                           batch_size=batch_size)

    #trainer.train(steps_per_epoch=50, epochs=5)
    trainer.apply_for_all_patches()

process_1_lopukangri()

