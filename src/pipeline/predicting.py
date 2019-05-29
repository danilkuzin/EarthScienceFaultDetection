from typing import List

import h5py
import numpy as np
import tensorflow as tf
import yaml

from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.DataPreprocessor.region_dataset import RegionDataset
from src.LearningKeras.train import KerasTrainer
from src.postprocessing.postprocessor import PostProcessor

from src.config import data_path

np.random.seed(1)
tf.set_random_seed(2)


def predict(datasets: List[int], model, models_folder, classes, channels, stride=25, batch_size=20):
    # if classes == 3:
    #     model = cnn_150x150x5_3class()
    # elif classes == 2:
    #     if len(channels) == 12:
    #         model = cnn_150x150x12()
    #     elif len(channels) == 11:
    #         model = cnn_150x150x11()
    #     elif len(channels) == 5:
    #         model = cnn_150x150x5()
    #     elif len(channels) == 4:
    #         model = cnn_150x150x4()
    #     elif len(channels) == 3:
    #         model = cnn_150x150x3()
    #     elif len(channels) == 1:
    #         model = cnn_150x150x1()
    #     else:
    #         raise Exception()
    # else:
    #     raise Exception('not supported')

    trainer = KerasTrainer(model=model)

    trainer.load(input_path=models_folder)

    for ind in datasets:
        data_preprocessor = RegionDataset(ind)

        boxes, probs = trainer.apply_for_sliding_window(
            data_preprocessor=data_preprocessor, patch_size=(150, 150), stride=stride, batch_size=batch_size,
            channels=channels)

        with h5py.File(f'{models_folder}/sliding_window_{ind}.h5', 'w') as hf:
            hf.create_dataset("boxes", data=boxes)
            hf.create_dataset("probs", data=probs)


def postprocess(datasets: List[int], models_folder, heatmap_mode="max"):

    for ind in datasets:

        data_preprocessor = RegionDataset(ind)

        gdal_backend = GdalBackend()
        with open(f"{data_path}/preprocessed/{ind}/gdal_params.yaml", 'r') as stream:
            gdal_params = yaml.safe_load(stream)
        gdal_backend.set_params(gdal_params['driver_name'], gdal_params['projection'],
                                     eval(gdal_params['geotransform']))

        with h5py.File(f'{models_folder}/sliding_window_{ind}.h5', 'r') as hf:
            boxes = hf["boxes"][:]
            probs = hf["probs"][:]

        original_2dimage_shape = (data_preprocessor.get_data_shape()[0], data_preprocessor.get_data_shape()[1])
        faults_postprocessor = PostProcessor(boxes=boxes, probs=probs[:, 0],
                                             original_2dimage_shape=original_2dimage_shape)
        res_faults = faults_postprocessor.heatmaps(mode=heatmap_mode)
        gdal_backend.write_image(
            f"{models_folder}/heatmaps_probs_{ind}",
            res_faults * 100)
        gdal_backend.write_surface(
            f"{models_folder}/heatmaps_faults_{ind}", res_faults)
