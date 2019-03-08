import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import logging

from src.DataPreprocessor.data_preprocessor import Mode, FeatureValue, DataPreprocessor
from src.DataPreprocessor.data_visualiser import DataVisualiser
from src.pipeline import global_params


def _plot_channels_with_features(data_preprocessor, data_visualiser, output_path):
    for key in data_preprocessor.channels.keys():
        data_visualiser.get_channel_with_feature_mask(key, opacity=90).save(output_path + f"features_{key}.tif")


def _plot_channels(data_preprocessor, data_visualiser, output_path):
    for key in data_preprocessor.channels.keys():
        data_visualiser.get_channel(key).save(output_path + f"{key}.tif")


def _plot_distributions(data_preprocessor:DataPreprocessor, output_path:str):
    nrows, ncols = 3, 5
    f, axis = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 20))
    ax_ind = 0
    for key in data_preprocessor.channels.keys():
        if data_preprocessor.channels[key].ndim == 3:
            for d in range(data_preprocessor.channels[key].shape[2]):
                ind_2d = np.unravel_index(ax_ind, (nrows, ncols))
                cur_ax = axis[ind_2d[0], ind_2d[1]]
                logging.info(f"plot distribution for {key}_{d}")
                sns.distplot(data_preprocessor.channels[key][:, :, d].flatten(), ax=cur_ax)
                cur_ax.set_title(f'{key}_{d}')
                ax_ind = ax_ind + 1
        elif data_preprocessor.channels[key].ndim == 2:
            ind_2d = np.unravel_index(ax_ind, (nrows, ncols))
            cur_ax = axis[ind_2d[0], ind_2d[1]]
            logging.info(f"plot distribution for {key}")
            sns.distplot(data_preprocessor.channels[key].flatten(), ax=cur_ax)
            cur_ax.set_title(f'{key}')
            ax_ind = ax_ind + 1
    plt.tight_layout()
    f.savefig(output_path + "features_distribution.png")
    f.clf()
    plt.close()

    f, axis = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 20))
    ax_ind = 0
    for key in data_preprocessor.normalised_channels.keys():
        if data_preprocessor.normalised_channels[key].ndim == 3:
            for d in range(data_preprocessor.normalised_channels[key].shape[2]):
                ind_2d = np.unravel_index(ax_ind, (nrows, ncols))
                cur_ax = axis[ind_2d[0], ind_2d[1]]
                logging.info(f"plot normalised distribution for {key}_{d}")
                sns.distplot(data_preprocessor.normalised_channels[key][:, :, d].flatten(), ax=cur_ax)
                cur_ax.set_title(f'{key}_{d}')
                ax_ind = ax_ind + 1
        elif data_preprocessor.normalised_channels[key].ndim == 2:
            ind_2d = np.unravel_index(ax_ind, (nrows, ncols))
            cur_ax = axis[ind_2d[0], ind_2d[1]]
            logging.info(f"plot normalised distribution for {key}")
            sns.distplot(data_preprocessor.normalised_channels[key].flatten(), ax=cur_ax)
            cur_ax.set_title(f'{key}')
            ax_ind = ax_ind + 1
    plt.tight_layout()
    f.savefig(output_path + "normalised_features_distribution.png")
    f.clf()
    plt.close()


def _plot_samples(num_patches, patch_size, bands, data_preprocessor, output_path):
    for lbl in [FeatureValue.FAULT, FeatureValue.FAULT_LOOKALIKE, FeatureValue.NONFAULT]:
        patches = np.zeros((num_patches, patch_size[0], patch_size[1], bands))
        for i in range(num_patches):
            patches[i] = data_preprocessor.sample_patch(label=lbl, patch_size=patch_size)

        for i in range(num_patches):
            cur_patch = patches[i]
            rgb, elevation, slope = data_preprocessor.denormalise(cur_patch)
            f, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(rgb)
            ax2.imshow(elevation)
            ax3.imshow(slope)
            f.tight_layout()
            f.savefig(output_path + "examples_{}_{}.png".format(lbl.name, i))
            f.clf()
            plt.close()


def visualise(datasets_ind, num_patches, patch_size, bands, plot_distributions):
    np.random.seed(1)
    tf.set_random_seed(2)

    datasets = [global_params.data_preprocessor_generators[i] for i in datasets_ind]
    for d_gen_ind, data_preprocessor_generator in enumerate(datasets):
        logging.info("init data preprocessors")
        if datasets_ind[d_gen_ind] in global_params.trainable:
            data_preprocessor = data_preprocessor_generator(Mode.TRAIN)
        else:
            data_preprocessor = data_preprocessor_generator(Mode.TEST)
        output_path = data_preprocessor.data_dir + "/visualisation/"
        pathlib.Path(output_path).mkdir(exist_ok=True)
        data_visualiser = DataVisualiser(data_preprocessor)

        logging.info("plot bands")
        if data_preprocessor.mode == Mode.TRAIN:
            _plot_channels_with_features(data_preprocessor, data_visualiser, output_path)
        elif data_preprocessor.mode == Mode.TEST:
            _plot_channels(data_preprocessor, data_visualiser, output_path)

        if plot_distributions:
            logging.info("plot distributions")
            _plot_distributions(data_preprocessor, output_path)

        if data_preprocessor.mode == Mode.TRAIN:
            logging.info("plot samples")
            _plot_samples(num_patches, patch_size, bands, data_preprocessor, output_path)

