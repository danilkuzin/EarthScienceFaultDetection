import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from src.DataPreprocessor.data_preprocessor import Mode, FeatureValue
from src.DataPreprocessor.data_visualiser import DataVisualiser
from src.pipeline import global_params

np.random.seed(1)
tf.set_random_seed(2)

num_patches = 7
patch_size = (150, 150)
bands = 5

for data_preprocessor_generator in global_params.data_preprocessor_generators_train:
    data_preprocessor = data_preprocessor_generator()
    output_path = data_preprocessor.data_dir + "/visualisation/"
    pathlib.Path(output_path).mkdir(exist_ok=True)
    data_visualiser = DataVisualiser(data_preprocessor)

    if data_preprocessor.mode == Mode.TRAIN:
        data_visualiser.get_optical_rgb_with_features_mask(opacity=90).save(output_path + "features_optical.tif")
        data_visualiser.get_elevation_with_features_mask(opacity=90).save(output_path + "features_elevation.tif")
        data_visualiser.get_slope_with_features_mask(opacity=90).save(output_path + "features_slope.tif")
        data_visualiser.get_nir_with_features_mask(opacity=90).save(output_path + "features_nir.tif")
        data_visualiser.get_ir_with_features_mask(opacity=90).save(output_path + "features_ir.tif")
        data_visualiser.get_swir1_with_features_mask(opacity=90).save(output_path + "features_swir1.tif")
        data_visualiser.get_swir2_with_features_mask(opacity=90).save(output_path + "features_swir2.tif")
        data_visualiser.get_panchromatic_with_features_mask(opacity=90).save(output_path + "features_panchromatic.tif")

    elif data_preprocessor.mode == Mode.TEST:
        data_visualiser.get_optical_rgb().save(output_path + "optical.tif")
        data_visualiser.get_elevation().save(output_path + "elevation.tif")
        data_visualiser.get_slope().save(output_path + "slope.tif")
        data_visualiser.get_nir().save(output_path + "nir.tif")
        data_visualiser.get_ir().save(output_path + "ir.tif")
        data_visualiser.get_swir1().save(output_path + "swir1.tif")
        data_visualiser.get_swir2().save(output_path + "swir2.tif")
        data_visualiser.get_panchromatic().save(output_path + "panchromatic.tif")

    f, axis = plt.subplots(2, 5, figsize=(30, 20))
    sns.distplot(data_preprocessor.channels['optical_rgb'][:, :, 0].flatten(), ax=axis[0, 0])
    sns.distplot(data_preprocessor.channels['optical_rgb'][:, :, 1].flatten(), ax=axis[0, 1])
    sns.distplot(data_preprocessor.channels['optical_rgb'][:, :, 2].flatten(), ax=axis[0, 2])
    sns.distplot(data_preprocessor.channels['elevation'].flatten(), ax=axis[0, 3])
    sns.distplot(data_preprocessor.channels['slope'].flatten(), ax=axis[0, 4])
    sns.distplot(data_preprocessor.channels['nir'].flatten(), ax=axis[1, 0])
    sns.distplot(data_preprocessor.channels['ir'].flatten(), ax=axis[1, 1])
    sns.distplot(data_preprocessor.channels['swir1'].flatten(), ax=axis[1, 2])
    sns.distplot(data_preprocessor.channels['swir2'].flatten(), ax=axis[1, 3])
    sns.distplot(data_preprocessor.channels['panchromatic'].flatten(), ax=axis[1, 4])

    plt.tight_layout()
    f.savefig(output_path + "features_distribution.png")
    f.clf()
    plt.close()

    _, axis = plt.subplots(2, 5, figsize=(30, 20))
    sns.distplot(data_preprocessor.normalised_channels['optical_rgb'][:, :, 0].flatten(), ax=axis[0, 0])
    sns.distplot(data_preprocessor.normalised_channels['optical_rgb'][:, :, 1].flatten(), ax=axis[0, 1])
    sns.distplot(data_preprocessor.normalised_channels['optical_rgb'][:, :, 2].flatten(), ax=axis[0, 2])
    sns.distplot(data_preprocessor.normalised_channels['elevation'].flatten(), ax=axis[0, 3])
    sns.distplot(data_preprocessor.normalised_channels['slope'].flatten(), ax=axis[0, 4])
    sns.distplot(data_preprocessor.normalised_channels['nir'].flatten(), ax=axis[1, 0])
    sns.distplot(data_preprocessor.normalised_channels['ir'].flatten(), ax=axis[1, 1])
    sns.distplot(data_preprocessor.normalised_channels['swir1'].flatten(), ax=axis[1, 2])
    sns.distplot(data_preprocessor.normalised_channels['swir2'].flatten(), ax=axis[1, 3])
    sns.distplot(data_preprocessor.normalised_channels['panchromatic'].flatten(), ax=axis[1, 4])
    plt.tight_layout()
    f.savefig(output_path + "normalised_features_distribution.png")
    f.clf()
    plt.close()

    if data_preprocessor.mode == Mode.TRAIN:
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
