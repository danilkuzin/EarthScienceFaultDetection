import logging

import h5py
import numpy as np
import yaml
from tqdm import trange

from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.DataPreprocessor.region_dataset import RegionDataset
from src.config import data_path


def visualise_training_patches(dataset):
    logging.info(f"loading data for dataset {dataset}")
    with h5py.File(f'../train_data/regions_{dataset}/data.h5', 'r') as hf:
        lbls = hf['lbls'][:]

    with h5py.File(f'../train_data/regions_{dataset}/data_coords.h5', 'r') as hf:
        coords = hf['coords'][:]

    coords = coords.astype(np.int)
    lbls = lbls.astype(np.int)
    logging.info(f"loaded data for dataset {dataset}, length={coords.shape[0]}")
    data_preprocessor = RegionDataset(dataset)
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

    gdal_backend = GdalBackend()
    with open(f"{data_path}/preprocessed/{dataset}/gdal_params.yaml", 'r') as stream:
        gdal_params = yaml.safe_load(stream)
    gdal_backend.set_params(gdal_params['driver_name'], gdal_params['projection'],
                                 eval(gdal_params['geotransform']))

    gdal_backend.write_surface(mask_0_path, mask_0)
    gdal_backend.write_surface(mask_1_path, mask_1)


    logging.info(f"images saved: {[mask_0_path, mask_1_path]}")

def visualise_training_patches_single_files(dataset, num):
    #todo estimate num from os
    data_preprocessor = RegionDataset(dataset)
    im_w, im_h, _ = data_preprocessor.get_data_shape()
    mask_0 = np.zeros((im_w, im_h))
    mask_1 = np.zeros((im_w, im_h))

    logging.info(f"loading data for dataset {dataset}")
    for n in range(num):
        with h5py.File(f'../../DataForEarthScienceFaultDetection/train_data/regions_{dataset}_single_files/data_{n}.h5', 'r') as hf:
            cur_lbl = hf['lbl'][:]
            cur_coords = hf['coord'][:]

        cur_coords = cur_coords.astype(np.int)
        cur_lbl = cur_lbl.astype(np.int)
        #logging.info(f"loaded data {n} for dataset {dataset}")

        if cur_lbl[0] == 1:
            mask_0[cur_coords[0]:cur_coords[1], cur_coords[2]:cur_coords[3]] = mask_0[cur_coords[0]:cur_coords[1], cur_coords[2]:cur_coords[3]] + 1
        elif cur_lbl[1] == 1:
            mask_1[cur_coords[0]:cur_coords[1], cur_coords[2]:cur_coords[3]] = mask_1[cur_coords[0]:cur_coords[1],
                                                                           cur_coords[2]:cur_coords[3]] + 1
        else:
            raise Exception()

    mask_0 = mask_0 / np.max(mask_0)
    mask_1 = mask_1 / np.max(mask_1)

    mask_0_path = f'../train_data/regions_{dataset}/data_patches_faults'
    mask_1_path = f'../train_data/regions_{dataset}/data_patches_non_faults'

    gdal_backend = GdalBackend()
    with open(f"{data_path}/preprocessed/{dataset}/gdal_params.yaml", 'r') as stream:
        gdal_params = yaml.safe_load(stream)
    gdal_backend.set_params(gdal_params['driver_name'], gdal_params['projection'],
                            eval(gdal_params['geotransform']))

    gdal_backend.write_surface(mask_0_path, mask_0)
    gdal_backend.write_surface(mask_1_path, mask_1)

    logging.info(f"images saved: {[mask_0_path, mask_1_path]}")

def visualise_training_patches_single_files_segmentation(dataset, num):
    data_preprocessor = RegionDataset(dataset)
    im_w, im_h, _ = data_preprocessor.get_data_shape()
    mask = .5 * np.ones((im_w, im_h))

    for n in trange(num):
        with h5py.File(f'../../DataForEarthScienceFaultDetection/train_data/regions_{dataset}_segmentation_mask/data_{n}.h5', 'r') as hf:
            cur_lbl = hf['lbl'][:]
            cur_coords = hf['coord'][:]

        cur_coords = cur_coords.astype(np.int)
        cur_lbl = cur_lbl.astype(np.int)

        mask[
            cur_coords[0]:cur_coords[1],
            cur_coords[2]:cur_coords[3]
        ] = cur_lbl

    mask_path = f'../train_data/regions_{dataset}/data_patches_segmentation'

    gdal_backend = GdalBackend()
    with open(f"{data_path}/preprocessed/{dataset}/gdal_params.yaml",
              'r') as stream:
        gdal_params = yaml.safe_load(stream)
    gdal_backend.set_params(gdal_params['driver_name'],
                            gdal_params['projection'],
                            eval(gdal_params['geotransform']))

    gdal_backend.write_surface(mask_path, mask)
