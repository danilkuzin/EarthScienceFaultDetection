import logging
import pathlib
import h5py

import sys

from tqdm import trange

sys.path.extend(['../../EarthScienceFaultDetection'])

from src.DataPreprocessor.data_generator import DataGenerator
from src.DataPreprocessor.data_preprocessor import Mode
from src.pipeline import global_params

def __generate_data_batch(data_generator, size):
    return data_generator.create_datasets(
        class_probabilities="two-class",
        patch_size=(150, 150),
        channels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        size=size,
        verbose=0)

def generate_data(datasets, size, data_batch_size=100):
    # todo currently data_batch_size is a multiplier of size
    if not size % data_batch_size == 0:
        raise ValueError()

    preprocessors = [global_params.data_preprocessor_generators[ind](Mode.TRAIN) for ind in datasets]
    for preprocessor, preprocessor_ind in zip(preprocessors, datasets):
        output_path = f"../train_data/regions_{preprocessor_ind}/"
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

        data_generator = DataGenerator(preprocessors=[preprocessor])

        imgs, lbls, coords = __generate_data_batch(data_generator, data_batch_size)
        with h5py.File(output_path + '/data.h5', 'w') as hf:
            hf.create_dataset("imgs", data=imgs, chunks=True, maxshape=(None, 150, 150, 10))
            hf.create_dataset("lbls", data=lbls, chunks=True, maxshape=(None, 2))

        with h5py.File(output_path + '/data_coords.h5', 'w') as hf:
            hf.create_dataset("coords", data=coords, chunks=True, maxshape=(None, 4))

        for i in trange(int(size/data_batch_size) - 1, desc="Iterating batches for data generation", initial=1):
            imgs, lbls, coords = __generate_data_batch(data_generator, data_batch_size)
            with h5py.File(output_path + '/data.h5', 'a') as hf:
                hf["imgs"].resize((hf["imgs"].shape[0] + imgs.shape[0]), axis=0)
                hf["imgs"][-imgs.shape[0]:] = imgs
                hf["lbls"].resize((hf["lbls"].shape[0] + lbls.shape[0]), axis=0)
                hf["lbls"][-lbls.shape[0]:] = lbls

            with h5py.File(output_path + '/data_coords.h5', 'a') as hf:
                hf["coords"].resize((hf["coords"].shape[0] + coords.shape[0]), axis=0)
                hf["coords"][-coords.shape[0]:] = coords

        logging.info(f"data saved: {[output_path + '/data.h5', output_path + '/data_coords.h5']}")








