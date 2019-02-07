import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import numpy as np
import tensorflow as tf

from src.DataPreprocessor.data_preprocessor import DataPreprocessor, Mode
from src.LearningKeras.net_architecture import cnn_150x150x5_3class
from src.LearningKeras.train import KerasTrainer
import h5py

from tqdm import trange

np.random.seed(1)
tf.set_random_seed(2)

from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.DataPreprocessor.data_visualiser import DataVisualiser

dataiobackend = GdalBackend()
data_preprocessor = DataPreprocessor(data_dir="../data/Region 1 - Lopukangri/",
                              backend=dataiobackend,
                              filename_prefix="tibet",
                              mode=Mode.TRAIN,
                              seed=1)

model_generator = lambda: cnn_150x150x5_3class()
ensemble_size = 3
batch_size = 10

trainer = KerasTrainer(model_generator=model_generator,
                       ensemble_size=ensemble_size,
                       data_preprocessor=data_preprocessor,
                       batch_size=batch_size)

trainer.models = []
for en in range(ensemble_size):
    model = model_generator()
    model.load_weights('../models_3class/model_{}.h5'.format(en))
    trainer.models.append(model)

dataiobackend_muggarboibo = GdalBackend()
data_preprocessor_muggarboibo = DataPreprocessor(data_dir="../data/Region 3 - Muggarboibo/",
                              backend=dataiobackend_muggarboibo,
                              filename_prefix="gyrc1",
                              mode=Mode.TEST,
                              seed=1)

boxes, avg_fault_probs, avg_lookalike_probs, avg_non_fault_probs = trainer.apply_for_sliding_window_3class_batch(data_preprocessor_muggarboibo, stride=50, batch_size=5)

with h5py.File('predicting_tmp_muggarboibo_3classes.h5', 'w') as hf:
    hf.create_dataset("dataset",  data=np.array(boxes))
with h5py.File('predicting_tmp2_muggarboibo_3classes.h5', 'w') as hf:
    hf.create_dataset("dataset",  data=np.array(avg_fault_probs))

# with h5py.File('predicting_tmp_muga_puruo.h5', 'r') as hf:
#     boxes = hf['dataset'][:]
# with h5py.File('predicting_tmp2_muga_puruo.h5', 'r') as hf:
#     avg_fault_probs = hf['dataset'][:]