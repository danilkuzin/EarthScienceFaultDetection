import numpy as np
import tensorflow as tf
import pathlib

from src.DataPreprocessor.data_generator import DataGenerator
from src.DataPreprocessor.data_preprocessor import DataPreprocessor, Mode
from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.LearningKeras.net_architecture import cnn_150x150x5_3class
from src.LearningKeras.train import KerasTrainer

np.random.seed(1)
tf.set_random_seed(2)

data_preprocessor_1 = DataPreprocessor(data_dir="../data/Region 1 - Lopukangri/",
                              backend=GdalBackend(),
                              filename_prefix="tibet",
                              mode=Mode.TRAIN,
                              seed=1)

data_preprocessor_2 = DataPreprocessor(data_dir="../data/Region 2 - Muga Puruo/",
                              backend=GdalBackend(),
                              filename_prefix="mpgr",
                              mode=Mode.TRAIN,
                              seed=1)

batch_size = 10

data_generator = DataGenerator(preprocessors=[data_preprocessor_1, data_preprocessor_2])
joint_generator = data_generator.generator(batch_size=batch_size,
                                           class_probabilities=np.array([1./3, 1./3, 1./3]),
                                           patch_size=(150, 150),
                                           channels=np.array([0, 1, 2, 3, 4]))

model_generator = lambda: cnn_150x150x5_3class()
ensemble_size = 2
batch_size = 5

trainer = KerasTrainer(model_generator=model_generator,
                       ensemble_size=ensemble_size,
                       batch_size=batch_size)

history_arr = trainer.train(steps_per_epoch=50, epochs=5, train_generator=joint_generator)

for i in range(ensemble_size):
    pathlib.Path('models_joint').mkdir(parents=True, exist_ok=True)
    trainer.models[i].save_weights('models_joint/model_{}.h5'.format(i))