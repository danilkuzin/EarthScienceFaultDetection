import numpy as np
import tensorflow as tf

from src.LearningKeras.net_architecture import cnn_150x150x5_3class, cnn_150x150x5, cnn_150x150x3, cnn_150x150x1
from src.LearningKeras.train import KerasTrainer
from src.pipeline import global_params
from src.postprocessing.postprocessor import PostProcessor

np.random.seed(1)
tf.set_random_seed(2)


def predict(models_folder, ensemble_size, classes, channels, heatmap_mode="max"):

    if classes == 3:
        model_generator = lambda: cnn_150x150x5_3class()
    elif classes == 2:
        if len(channels) == 5:
            model_generator = lambda: cnn_150x150x5()
        elif len(channels) == 3:
            model_generator = lambda: cnn_150x150x3()
        elif len(channels) == 1:
            model_generator = lambda: cnn_150x150x1()
        else:
            raise Exception()
    else:
        raise Exception('not supported')

    trainer = KerasTrainer(model_generator=model_generator,
                           ensemble_size=ensemble_size)

    trainer.load(input_path=models_folder)

    for (preprocessor_ind, data_preprocessor_generator) in enumerate(global_params.data_preprocessor_generators_test):
        data_preprocessor = data_preprocessor_generator()
        boxes, probs = trainer.apply_for_sliding_window(
            data_preprocessor=data_preprocessor, patch_size=(150, 150), stride=50, batch_size=20, channels=channels)
        original_2dimage_shape = (data_preprocessor.optical_rgb.shape[0], data_preprocessor.optical_rgb.shape[1])
        faults_postprocessor = PostProcessor(boxes=boxes, probs=probs[:, 0],
                                             original_2dimage_shape=original_2dimage_shape)
        res_faults = faults_postprocessor.heatmaps(mode=heatmap_mode)
        data_preprocessor.data_io_backend.write_surface("heatmaps_faults_{}.tif".format(preprocessor_ind), res_faults)

        if classes == 3:
            lookalikes_postprocessor = PostProcessor(boxes=boxes, probs=probs[:, 1],
                                                     original_2dimage_shape=original_2dimage_shape)
            res_lookalikes = lookalikes_postprocessor.heatmaps(mode=heatmap_mode)
            data_preprocessor.data_io_backend.write_surface("heatmaps_lookalikes_{}.tif".format(preprocessor_ind),
                                                    res_lookalikes)

            nonfaults_postprocessor = PostProcessor(boxes=boxes, probs=probs[:, 2],
                                                    original_2dimage_shape=original_2dimage_shape)
            res_nonfaults = nonfaults_postprocessor.heatmaps(mode="max")
            data_preprocessor.data_io_backend.write_surface("heatmaps_nonfaults_{}.tif".format(preprocessor_ind),
                                                    res_nonfaults)

        elif classes == 2:
            nonfaults_postprocessor = PostProcessor(boxes=boxes, probs=probs[:, 1],
                                                    original_2dimage_shape=original_2dimage_shape)
            res_nonfaults = nonfaults_postprocessor.heatmaps(mode=heatmap_mode)
            data_preprocessor.data_io_backend.write_surface("heatmaps_nonfaults_{}.tif".format(preprocessor_ind),
                                                    res_nonfaults)
