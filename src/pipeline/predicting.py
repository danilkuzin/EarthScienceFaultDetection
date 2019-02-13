import numpy as np
import tensorflow as tf

from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.DataPreprocessor.data_preprocessor import DataPreprocessor, Mode
from src.LearningKeras.net_architecture import cnn_150x150x5_3class, cnn_150x150x5_2class_3convolutions
from src.LearningKeras.train import KerasTrainer
from src.postprocessing.postprocessor import PostProcessor

np.random.seed(1)
tf.set_random_seed(2)


def predict(models_folder, ensemble_size, classes):
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

    if classes == 3:
        model_generator = lambda: cnn_150x150x5_3class()
    elif classes == 2:
        model_generator = lambda: cnn_150x150x5_2class_3convolutions()
    else:
        raise Exception('not supported')

    trainer = KerasTrainer(model_generator=model_generator,
                           ensemble_size=ensemble_size)

    trainer.load(input_path=models_folder)

    for (preprocessor_ind, data_preprocessor) in enumerate(
            [data_preprocessor_1, data_preprocessor_2, data_preprocessor_3]):
        boxes, probs = trainer.apply_for_sliding_window(
            data_preprocessor=data_preprocessor, patch_size=(150, 150), stride=50, batch_size=20)
        original_2dimage_shape = (data_preprocessor.optical_rgb.shape[0], data_preprocessor.optical_rgb.shape[1])
        faults_postprocessor = PostProcessor(boxes=boxes, probs=probs[:, 0],
                                             original_2dimage_shape=original_2dimage_shape)
        res_faults = faults_postprocessor.heatmaps(mode="max")
        data_preprocessor.backend.write_surface("heatmaps_faults_{}.tif".format(preprocessor_ind), res_faults)

        if classes == 3:
            lookalikes_postprocessor = PostProcessor(boxes=boxes, probs=probs[:, 1],
                                                     original_2dimage_shape=original_2dimage_shape)
            res_lookalikes = lookalikes_postprocessor.heatmaps(mode="max")
            data_preprocessor.backend.write_surface("heatmaps_lookalikes_{}.tif".format(preprocessor_ind),
                                                    res_lookalikes)

            nonfaults_postprocessor = PostProcessor(boxes=boxes, probs=probs[:, 2],
                                                    original_2dimage_shape=original_2dimage_shape)
            res_nonfaults = nonfaults_postprocessor.heatmaps(mode="max")
            data_preprocessor.backend.write_surface("heatmaps_nonfaults_{}.tif".format(preprocessor_ind),
                                                    res_nonfaults)

        elif classes == 2:
            nonfaults_postprocessor = PostProcessor(boxes=boxes, probs=probs[:, 1],
                                                    original_2dimage_shape=original_2dimage_shape)
            res_nonfaults = nonfaults_postprocessor.heatmaps(mode="max")
            data_preprocessor.backend.write_surface("heatmaps_nonfaults_{}.tif".format(preprocessor_ind),
                                                    res_nonfaults)
