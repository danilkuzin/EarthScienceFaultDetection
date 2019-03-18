import numpy as np
import tensorflow as tf

from src.DataPreprocessor.data_preprocessor import Mode
from src.LearningKeras.net_architecture import cnn_150x150x3, cnn_150x150x5, alexnet
from src.pipeline import global_params
import h5py
import matplotlib.pyplot as plt

np.random.seed(1)
tf.set_random_seed(2)

class FeatureSelector:
    def __init__(self, channels = np.array([0, 1, 2]), test_size = 500):
        self.train_data_preprocessor = global_params.data_preprocessor_generators[0](Mode.TRAIN)
        self.test_data_preprocessor = global_params.data_preprocessor_generators[1](Mode.TRAIN)
        self.validation_data_preprocessor = global_params.data_preprocessor_generators[1](Mode.TRAIN)
        self.model = cnn_150x150x3()

    def prepare_datasets(self):
        img_test, lbl_test = self.test_data_preprocessor.get_sample_2class_lookalikes_with_nonfaults(batch_size=50,
                                                                                            class_probabilities=np.array([0.5, 0.25, 0.25]),
                                                                                            patch_size=(150, 150),
                                                                                            channels=channels)                                                                                         channels=channels)
        train_generator = train_data_preprocessor.train_generator_2class_lookalikes_with_nonfaults(batch_size=50,
                                                                                                    class_probabilities=np.array([0.5, 0.25, 0.25]),
                                                                                                    patch_size=(150, 150),
                                                                                                    channels=channels)

        validation_generator = validation_data_preprocessor.train_generator_2class_lookalikes_with_nonfaults(batch_size=50,
    class_probabilities = np.array([0.5, 0.25, 0.25]), patch_size = (224, 224), channels = channels)

    def check_preformace()
# num_epochs = 50
# for ep in range(num_epochs):
#     model.fit_generator(generator=train_generator,
#                               steps_per_epoch=10,
#                               epochs=1,
#                               validation_data=train_generator,
#                               validation_steps=5,
#                               workers=0,
#                               use_multiprocessing=False)
#     pred_lbls = np.argmax(model.predict(img_test), axis=1)
#
#     quality_test.append(np.sum(pred_lbls == test_lbls)/lbl_test.shape[0])
#
# with open(f"test_quality_{''.join(str(i) for i in channels.tolist())}.txt", "w") as text_file:
#     print(f"Channels: {channels}, Quality: {quality_test} \n", file=text_file)

# with validation in the 2nd dataset:



history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=20,
                              epochs=100,
                              validation_data=validation_generator,
                              validation_steps=2,
                              workers=0,
                              use_multiprocessing=False)

with h5py.File('train_valid_hist.h5', 'w') as hf:
        hf.create_dataset("history_val_loss", data=history.history['val_loss'])
        hf.create_dataset("history_val_acc", data=history.history['val_acc'])
        hf.create_dataset("loss", data=history.history['loss'])
        hf.create_dataset("acc", data=history.history['acc'])
# hf.create_dataset("params", data=history.params)

    def plot(self):
        with h5py.File('train_valid_hist.h5', 'r') as hf:
            val_loss = hf['history_val_loss'][:]
            val_acc = hf['history_val_acc'][:]
            loss = hf['loss'][:]
            acc = hf['acc'][:]

        plt.plot(acc)
        plt.plot(val_acc)
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        plt.plot(loss)
        plt.plot(val_loss)
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()