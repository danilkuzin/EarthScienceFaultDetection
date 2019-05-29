import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import logging

from src.DataPreprocessor.region_normalised_visualiser import RegionNormalisedVisualiser
from src.DataPreprocessor.region_raw_visualiser import RegionRawVisualiser





#

#
#
# def _plot_samples(num_patches, patch_size, bands, data_preprocessor, output_path):
#     for lbl in [FeatureValue.FAULT, FeatureValue.FAULT_LOOKALIKE, FeatureValue.NONFAULT]:
#         patches = np.zeros((num_patches, patch_size[0], patch_size[1], bands))
#         for i in range(num_patches):
#             patches[i], coords = data_preprocessor.sample_patch(label=lbl.value, patch_size=patch_size)
#
#         for i in range(num_patches):
#             cur_patch = patches[i]
#             rgb, elevation, slope = data_preprocessor.denormalise(cur_patch)
#             f, (ax1, ax2, ax3) = plt.subplots(1, 3)
#             ax1.imshow(rgb)
#             ax2.imshow(elevation)
#             ax3.imshow(slope)
#             f.tight_layout()
#             f.savefig(output_path + f"examples_{lbl.name}_{i}.png")
#             f.clf()
#             plt.close()


def visualise(datasets_ind, num_patches, patch_size, bands, plot_distributions, inp_output_path, crop=None):
    np.random.seed(1)
    tf.set_random_seed(2)

    for d_ind in datasets_ind:
        logging.info("init data preprocessor")
        region_raw_visualiser = RegionRawVisualiser(region_id=d_ind)
        region_raw_visualiser.write_features()
        region_raw_visualiser.plot_channels_with_features(crop=crop)
        region_raw_visualiser.plot_channels(crop=crop)

        region_normalised_visualiser = RegionNormalisedVisualiser(region_id=d_ind)

        if plot_distributions:
            logging.info("plot distributions")
            region_raw_visualiser.plot_distributions()
            region_normalised_visualiser.plot_distributions()

        # if region_dataset.trainable:
        #     logging.info("plot samples")
        #     _plot_samples(num_patches, patch_size, bands, region_dataset, output_path)


