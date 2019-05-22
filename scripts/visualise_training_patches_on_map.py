import logging

import h5py
import numpy as np
from tqdm import trange

from src.DataPreprocessor.data_preprocessor import Mode
from src.pipeline import global_params


def visualise_training_patches(dataset):
    logging.info(f"loading data for dataset {dataset}")
    with h5py.File(f'../train_data/regions_{dataset}/data.h5', 'r') as hf:
        lbls = hf['lbls'][:]

    with h5py.File(f'../train_data/regions_{dataset}/data_coords.h5', 'r') as hf:
        coords = hf['coords'][:]

    coords = coords.astype(np.int)
    lbls = lbls.astype(np.int)
    logging.info(f"loaded data for dataset {dataset}, length={coords.shape[0]}")
    data_preprocessor = global_params.data_preprocessor_generators[dataset](Mode.TRAIN)
    im_w, im_h, _ = data_preprocessor.get_data_shape()

    mask_0 = np.zeros((im_w, im_h))
    mask_1 = np.zeros((im_w, im_h))
    for n in trange(coords.shape[0]):
        cur_coors = coords[n]
        cur_lbl = lbls[n]

        if cur_lbl[0] == 1:
            mask_0[cur_coors[0]:cur_coors[1], cur_coors[2]:cur_coors[3]] = mask_0[cur_coors[0]:cur_coors[1], cur_coors[2]:cur_coors[3]] + 1
        elif cur_lbl[1] == 1:
            mask_1[cur_coors[0]:cur_coors[1], cur_coors[2]:cur_coors[3]] = mask_1[cur_coors[0]:cur_coors[1],
                                                                       cur_coors[2]:cur_coors[3]] + 1
        else:
            raise Exception()

    mask_0 = mask_0 / np.max(mask_0)
    mask_1 = mask_1 / np.max(mask_1)

    mask_0_path = f'../train_data/regions_{dataset}/data_patches_faults'
    mask_1_path = f'../train_data/regions_{dataset}/data_patches_non_faults'
    data_preprocessor.data_io_backend.write_surface(mask_0_path, mask_0)
    data_preprocessor.data_io_backend.write_surface(mask_1_path, mask_1)

    logging.info(f"images saved: {[mask_0_path, mask_1_path]}")




