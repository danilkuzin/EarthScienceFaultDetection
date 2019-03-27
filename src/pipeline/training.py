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
    cnn_150x150x3, cnn_150x150x1, cnn_150x150x12, cnn_150x150x11, cnn_150x150x4, cnn_150x150x1_3class, \
    cnn_150x150x3_3class, cnn_150x150x10
#from src.LearningKeras.net_architecture import CnnModel150x150x5
from src.LearningKeras.train import KerasTrainer

# use Pipeline instead
from src.pipeline import global_params


#todo sample validation set at beginning, once
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

    if class_probabilities == "equal":
        class_probabilities_int = np.array([1. / 3, 1. / 3, 1. / 3])
        train_joint_generator = train_data_generator.generator_3class(batch_size=batch_size,
                                                   class_probabilities=class_probabilities_int,
                                                   patch_size=patch_size,
                                                   channels=np.array(channels))
        valid_joint_generator = valid_data_generator.generator_3class(batch_size=valid_size,
                                                                      class_probabilities=class_probabilities_int,
                                                                      patch_size=patch_size,
                                                                      channels=np.array(channels))

        if len(channels) == 5:
            model = cnn_150x150x5_3class()
        elif len(channels) == 3:
            model = cnn_150x150x3_3class()
        elif len(channels) == 1:
            model = cnn_150x150x1_3class()
        else:
            raise Exception()

    elif class_probabilities == "two-class":
        class_probabilities_int = np.array([0.5, 0.25, 0.25])
        train_joint_generator = train_data_generator.generator_2class_lookalikes_with_nonfaults(batch_size=batch_size,
                                                   class_probabilities=class_probabilities_int,
                                                   patch_size=patch_size,
                                                   channels=np.array(channels))
        valid_joint_generator = valid_data_generator.generator_2class_lookalikes_with_nonfaults(batch_size=valid_size,
                                                                                                class_probabilities=class_probabilities_int,
                                                                                                patch_size=patch_size,
                                                                                                channels=np.array(
                                                                                                    channels))
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

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=32, write_graph=True,
                                       write_grads=True, write_images=True),
        tf.keras.callbacks.CSVLogger(filename='log.csv', separator=',', append=False)
    ]

    imgs_valid, lbls_valid, _ = next(valid_joint_generator)
    valid_joint_generator = None # release resources #todo replace with "with"
    valid_preprocessors = None # release resources

    #todo train_joint_generator may now return 3 arrays instead of two, which is incorrect for fit_generator
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

    #switched to tensorboard + csv
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



