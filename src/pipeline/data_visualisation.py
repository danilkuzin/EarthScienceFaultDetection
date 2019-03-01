import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from src.DataPreprocessor.data_preprocessor import Mode, FeatureValue
from src.DataPreprocessor.data_visualiser import DataVisualiser
from src.pipeline import global_params


#todo iterate over keys, not manually
def visualise(datasets, num_patches, patch_size, bands, plot_distributions):
    np.random.seed(1)
    tf.set_random_seed(2)

    datasets = [global_params.data_preprocessor_generators[i] for i in datasets]
    for d_gen_ind, data_preprocessor_generator in enumerate(datasets):
        if datasets[d_gen_ind] in global_params.trainable:
            data_preprocessor = data_preprocessor_generator(Mode.TRAIN)
        else:
            data_preprocessor = data_preprocessor_generator(Mode.TEST)
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
            data_visualiser.get_curve_with_features_mask(opacity=90).save(output_path + "features_curve.tif")
            data_visualiser.get_erosion_with_features_mask(opacity=90).save(output_path + "features_erosion.tif")

        elif data_preprocessor.mode == Mode.TEST:
            data_visualiser.get_optical_rgb().save(output_path + "optical.tif")
            data_visualiser.get_elevation().save(output_path + "elevation.tif")
            data_visualiser.get_slope().save(output_path + "slope.tif")
            data_visualiser.get_nir().save(output_path + "nir.tif")
            data_visualiser.get_ir().save(output_path + "ir.tif")
            data_visualiser.get_swir1().save(output_path + "swir1.tif")
            data_visualiser.get_swir2().save(output_path + "swir2.tif")
            data_visualiser.get_panchromatic().save(output_path + "panchromatic.tif")
            data_visualiser.get_curve().save(output_path + "curve.tif")
            data_visualiser.get_erosion().save(output_path + "erosion.tif")

        if plot_distributions:
            f, axis = plt.subplots(3, 5, figsize=(30, 20))
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
            sns.distplot(data_preprocessor.channels['curve'].flatten(), ax=axis[2, 0])
            sns.distplot(data_preprocessor.channels['erosion'].flatten(), ax=axis[2, 1])
            plt.tight_layout()
            f.savefig(output_path + "features_distribution.png")
            f.clf()
            plt.close()

            f, axis = plt.subplots(3, 5, figsize=(30, 20))
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
            sns.distplot(data_preprocessor.normalised_channels['curve'].flatten(), ax=axis[2, 0])
            sns.distplot(data_preprocessor.normalised_channels['erosion'].flatten(), ax=axis[2, 1])
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
