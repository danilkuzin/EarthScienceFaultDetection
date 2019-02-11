import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import h5py

from src.DataPreprocessor.data_preprocessor import DataPreprocessor, Mode
from src.LearningKeras.net_architecture import cnn_150x150x5_3class
from src.LearningKeras.train import KerasTrainer
from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend

np.random.seed(1)
tf.set_random_seed(2)

data_preprocessor_1 = DataPreprocessor(data_dir="../data/Region 1 - Lopukangri/",
                              backend=GdalBackend(),
                              filename_prefix="tibet",
                              mode=Mode.TEST,
                              seed=1)

data_preprocessor_2 = DataPreprocessor(data_dir="../data/Region 2 - Muga Puruo/",
                              backend=GdalBackend(),
                              filename_prefix="mpgr",
                              mode=Mode.TEST,
                              seed=1)

data_preprocessor_3 = DataPreprocessor(data_dir="../data/Region 3 - Muggarboibo/",
                     backend=GdalBackend(),
                     filename_prefix="gyrc1",
                     mode=Mode.TEST,
                     seed=1)


model_generator = lambda: cnn_150x150x5_3class()
ensemble_size = 5
batch_size = 5

trainer = KerasTrainer(model_generator=model_generator,
                       ensemble_size=ensemble_size,
                       batch_size=batch_size)

for i in range(ensemble_size):
    model = model_generator()
    model.load_weights('models_joint/model_{}.h5'.format(i))
    trainer.models.append(model)

for (preprocesor_ind, data_preprocessor) in enumerate([data_preprocessor_1, data_preprocessor_2, data_preprocessor_3]):
    boxes, avg_fault_probs, avg_lookalike_probs, avg_non_fault_probs = trainer.apply_for_sliding_window_3class_batch(
        data_preprocessor, stride=50, batch_size=20)

    res_faults = trainer.apply_for_sliding_window_heatmaps(boxes, avg_fault_probs, data_preprocessor)

    cmap = plt.get_cmap('jet')
    rgba_img_faults = cmap(res_faults)
    rgb_img_faults = np.delete(rgba_img_faults, 3, 2)
    rgb_img_faults=(rgb_img_faults[:, :, :3] * 255).astype(np.uint8)

    data_preprocessor.backend.write_image("heatmaps_3class_faults_{}.tif".format(preprocesor_ind), rgb_img_faults)

    res_lookalikes = trainer.apply_for_sliding_window_heatmaps(boxes, avg_lookalike_probs, data_preprocessor)

    cmap = plt.get_cmap('jet')
    rgba_img_lookalikes = cmap(res_lookalikes)
    rgb_img_lookalikes = np.delete(rgba_img_lookalikes, 3, 2)
    rgb_img_lookalikes=(rgb_img_lookalikes[:, :, :3] * 255).astype(np.uint8)
    data_preprocessor.backend.write_image("heatmaps_3class_lookalikes_{}.tif".format(preprocesor_ind), rgb_img_lookalikes)
