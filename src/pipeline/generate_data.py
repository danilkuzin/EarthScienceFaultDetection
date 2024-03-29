import logging
import pathlib
import h5py

import sys

from tqdm import trange

sys.path.extend(['../../EarthScienceFaultDetection'])

from src.DataPreprocessor.region_dataset import RegionDataset
from src.DataPreprocessor.data_generator import DataGenerator
import src.config

def __generate_data_batch(data_generator, size, class_probabilities):
    return data_generator.create_datasets(
        class_probabilities=class_probabilities,
        patch_size=(150, 150),
        channels=[0, 1, 2, 3, 4],
        size=size,
        verbose=0)

def generate_data(dataset_inds, size, data_batch_size=100, lookalike_ratio=None):
    # todo currently data_batch_size is a multiplier of size
    if not size % data_batch_size == 0:
        raise ValueError()

    if lookalike_ratio is None:
        lookalike_ratio = [0.25] * len(dataset_inds)

    for i in range(len(lookalike_ratio)):
        if lookalike_ratio[i] is None:
            lookalike_ratio[i] = 0.25

    region_datasets = [RegionDataset(ind) for ind in dataset_inds]
    for dataset, preprocessor_ind, lookalike_ratio_for_dataset in zip(region_datasets, dataset_inds, lookalike_ratio):
        output_path = f"../train_data/regions_{preprocessor_ind}/"
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

        data_generator = DataGenerator(preprocessors=[dataset])
        class_probabilities = [0.5, lookalike_ratio_for_dataset, 0.5 - lookalike_ratio_for_dataset]

        imgs, lbls, coords = __generate_data_batch(data_generator, data_batch_size, class_probabilities)
        with h5py.File(output_path + '/data.h5', 'w') as hf:
            hf.create_dataset("imgs", data=imgs, chunks=True, maxshape=(None, 150, 150, 10))
            hf.create_dataset("lbls", data=lbls, chunks=True, maxshape=(None, 2))

        with h5py.File(output_path + '/data_coords.h5', 'w') as hf:
            hf.create_dataset("coords", data=coords, chunks=True, maxshape=(None, 4))

        for i in trange(int(size/data_batch_size) - 1, desc="Iterating batches for data generation", initial=1):
            imgs, lbls, coords = __generate_data_batch(data_generator, data_batch_size, class_probabilities)
            with h5py.File(output_path + '/data.h5', 'a') as hf:
                hf["imgs"].resize((hf["imgs"].shape[0] + imgs.shape[0]), axis=0)
                hf["imgs"][-imgs.shape[0]:] = imgs
                hf["lbls"].resize((hf["lbls"].shape[0] + lbls.shape[0]), axis=0)
                hf["lbls"][-lbls.shape[0]:] = lbls

            with h5py.File(output_path + '/data_coords.h5', 'a') as hf:
                hf["coords"].resize((hf["coords"].shape[0] + coords.shape[0]), axis=0)
                hf["coords"][-coords.shape[0]:] = coords

        logging.info(f"data saved: {[output_path + '/data.h5', output_path + '/data_coords.h5']}")


def __generate_and_save_data_batch(data_generator, size, class_probabilities, output_path, n):
    imgs, lbls, coords = __generate_data_batch(data_generator, size, class_probabilities)
    with h5py.File(f'{output_path}/data_{n}.h5', 'w') as hf:
        hf.create_dataset("img", data=imgs[0])
        hf.create_dataset("lbl", data=lbls[0])
        hf.create_dataset("coord", data=coords[0])


def generate_data_single_files(dataset_inds, size, lookalike_ratio=None):
    if lookalike_ratio is None:
        lookalike_ratio = [0.25] * len(dataset_inds)

    for i in range(len(lookalike_ratio)):
        if lookalike_ratio[i] is None:
            lookalike_ratio[i] = 0.25

    region_datasets = [RegionDataset(ind) for ind in dataset_inds]

    for dataset, dataset_ind, lookalike_ratio_for_dataset in zip(region_datasets, dataset_inds, lookalike_ratio):
        output_path = f"../../DataForEarthScienceFaultDetection/train_data/regions_{dataset_ind}_single_files/"
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

        data_generator = DataGenerator(preprocessors=[dataset])
        class_probabilities = [0.5, lookalike_ratio_for_dataset, 0.5 - lookalike_ratio_for_dataset]

        for n in trange(size):
            __generate_and_save_data_batch(data_generator, 1, class_probabilities, output_path, n)
        # Parallel(n_jobs=4)(delayed(__generate_and_save_data_batch)(data_generator, 1, class_probabilities, output_path, n)
        #                    for n in range(size))

        logging.info(f"data saved: {output_path}")










