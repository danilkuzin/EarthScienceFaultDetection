from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pathlib

from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.DataPreprocessor.PatchesOutputBackend.in_memory_backend import InMemoryBackend
from src.DataPreprocessor.data_generator import DataGenerator
from src.DataPreprocessor.data_preprocessor import DataPreprocessor, Mode
from src.LearningKeras.net_architecture import cnn_150x150x5_3class, cnn_150x150x5_2class_3convolutions, cnn_150x150x5, \
    cnn_150x150x3, cnn_150x150x1
from src.LearningKeras.train import KerasTrainer

# use Pipeline instead
def train(train_datasets: List[int], class_probabilities: str, batch_size: int, patch_size: Tuple[int, int],
          channels: List[int], ensemble_size: int, train_lib="keras", output_path=""):
    np.random.seed(1)
    tf.set_random_seed(2)

    preprocessors = []
    if 1 in train_datasets:
        preprocessors.append(DataPreprocessor(
            data_dir="../data/Region 1 - Lopukangri/",
            data_io_backend=GdalBackend(),
            patches_output_backend=InMemoryBackend(),
            filename_prefix="tibet",
            mode=Mode.TRAIN,
            seed=1)
        )

    if 2 in train_datasets:
        preprocessors.append(DataPreprocessor(
            data_dir="../data/Region 2 - Muga Puruo/",
            data_io_backend=GdalBackend(),
            patches_output_backend=InMemoryBackend(),
            filename_prefix="mpgr",
            mode=Mode.TRAIN,
            seed=1)
        )

    data_generator = DataGenerator(preprocessors=preprocessors)

    if class_probabilities == "equal":
        class_probabilities_int = np.array([1. / 3, 1. / 3, 1. / 3])
        joint_generator = data_generator.generator_3class(batch_size=batch_size,
                                                   class_probabilities=class_probabilities_int,
                                                   patch_size=patch_size,
                                                   channels=np.array(channels))
        if train_lib == "keras":
            trainer = KerasTrainer(model_generator=lambda: cnn_150x150x5_3class(),
                                   ensemble_size=ensemble_size)
        elif train_lib == "tensorflow":
            trainer = KerasTrainer(model_generator=lambda: cnn_150x150x5_3class(),
                                   ensemble_size=ensemble_size)
    elif class_probabilities == "two-class":
        class_probabilities_int = np.array([0.5, 0.25, 0.25])
        joint_generator = data_generator.generator_2class_lookalikes_with_nonfaults(batch_size=batch_size,
                                                   class_probabilities=class_probabilities_int,
                                                   patch_size=patch_size,
                                                   channels=np.array(channels))
        if len(channels) == 5:
            model_generator = lambda: cnn_150x150x5()
        elif len(channels) == 3:
            model_generator = lambda: cnn_150x150x3()
        elif len(channels) == 1:
            model_generator = lambda: cnn_150x150x1()
        else:
            raise Exception()
        trainer = KerasTrainer(model_generator=model_generator,
                               ensemble_size=ensemble_size)
    else:
        raise Exception('Not implemented')

    history_arr = trainer.train(steps_per_epoch=100, epochs=10, train_generator=joint_generator)

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    trainer.save(output_path='{}trained_models_{}'.format(output_path, ''.join(str(i) for i in train_datasets)))

    for (hist_ind, history) in enumerate(history_arr):
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig("Model accuracy_{}.png".format(hist_ind))
        plt.close()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig("Model loss_{}.png".format(hist_ind))
        plt.close()



