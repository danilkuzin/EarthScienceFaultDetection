from typing import List

import h5py
import numpy as np
import tensorflow as tf

from src.DataPreprocessor.data_preprocessor import Mode
from src.LearningKeras.net_architecture import cnn_150x150x5_3class, cnn_150x150x5, cnn_150x150x3, cnn_150x150x1, \
    cnn_150x150x12, cnn_150x150x11, cnn_150x150x4
from src.LearningKeras.train import KerasTrainer
from src.pipeline import global_params
from src.postprocessing.postprocessor import PostProcessor

np.random.seed(1)
tf.set_random_seed(2)


def predict(datasets: List[int], models_folder, classes, channels, stride=25, batch_size=20):
    if classes == 3:
        model = cnn_150x150x5_3class()
    elif classes == 2:
        if len(channels) == 12:
            model = cnn_150x150x12()
        elif len(channels) == 11:
            model = cnn_150x150x11()
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
        raise Exception('not supported')

    trainer = KerasTrainer(model=model)

    trainer.load(input_path=models_folder)

    for (preprocessor_ind, data_preprocessor_generator) in enumerate(global_params.data_preprocessor_generators):
        if preprocessor_ind not in datasets:
            continue

        data_preprocessor = data_preprocessor_generator(Mode.TEST)
        boxes, probs = trainer.apply_for_sliding_window(
            data_preprocessor=data_preprocessor, patch_size=(150, 150), stride=stride, batch_size=batch_size,
            channels=channels)

        with h5py.File(f'{models_folder}/sliding_window_{preprocessor_ind}.h5', 'w') as hf:
            hf.create_dataset("boxes", data=boxes)
            hf.create_dataset("probs", data=probs)


def postprocess(datasets: List[int], models_folder, heatmap_mode="max"):

    for (preprocessor_ind, data_preprocessor_generator) in enumerate(global_params.data_preprocessor_generators):
        if preprocessor_ind not in datasets:
            continue

        data_preprocessor = data_preprocessor_generator(Mode.TEST)

        with h5py.File(f'{models_folder}/sliding_window_{preprocessor_ind}.h5', 'r') as hf:
            boxes = hf["boxes"][:]
            probs = hf["probs"][:]

        original_2dimage_shape = (data_preprocessor.get_data_shape()[0], data_preprocessor.get_data_shape()[1])
        faults_postprocessor = PostProcessor(boxes=boxes, probs=probs[:, 0],
                                             original_2dimage_shape=original_2dimage_shape)
        res_faults = faults_postprocessor.heatmaps(mode=heatmap_mode)
        data_preprocessor.data_io_backend.write_image(
            f"{models_folder}/heatmaps_probs_{preprocessor_ind}",
            res_faults * 100)
        data_preprocessor.data_io_backend.write_surface(
            f"{models_folder}/heatmaps_faults_{preprocessor_ind}", res_faults)
