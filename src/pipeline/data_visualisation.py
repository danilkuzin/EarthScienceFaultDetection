import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import pathlib

from src.DataPreprocessor.data_preprocessor import DataPreprocessor, Mode, FeatureValue
from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.DataPreprocessor.data_visualiser import DataVisualiser

np.random.seed(1)
tf.set_random_seed(2)

data_preprocessors = [
    DataPreprocessor(data_dir="../data/Region 1 - Lopukangri/",
                     backend=GdalBackend(),
                     filename_prefix="tibet",
                     mode=Mode.TRAIN,
                     seed=1),
    DataPreprocessor(data_dir="../data/Region 2 - Muga Puruo/",
                     backend=GdalBackend(),
                     filename_prefix="mpgr",
                     mode=Mode.TRAIN,
                     seed=1),
    DataPreprocessor(data_dir="../data/Region 3 - Muggarboibo/",
                     backend=GdalBackend(),
                     filename_prefix="gyrc1",
                     mode=Mode.TEST,
                     seed=1)
]

num_patches = 7
patch_size = (150, 150)
bands = 5

for data_preprocessor in data_preprocessors:
    output_path = data_preprocessor.data_dir + "/visualisation/"
    pathlib.Path(output_path).mkdir(exist_ok=True)
    data_visualiser = DataVisualiser(data_preprocessor)

    if data_preprocessor.mode == Mode.TRAIN:
        data_visualiser.get_optical_rgb_with_features_mask(opacity=90).save(output_path + "features_optical.tif")
        data_visualiser.get_elevation_with_features_mask(opacity=90).save(output_path + "features_elevation.tif")
        data_visualiser.get_slope_with_features_mask(opacity=90).save(output_path + "features_slope.tif")
    elif data_preprocessor.mode == Mode.TEST:
        data_visualiser.get_optical_rgb().save(output_path + "optical.tif")
        data_visualiser.get_elevation().save(output_path + "elevation.tif")
        data_visualiser.get_slope().save(output_path + "slope.tif")

    f, axis = plt.subplots(1, 5)
    sns.distplot(data_preprocessor.optical_rgb[:, :, 0].flatten(), ax=axis[0])
    sns.distplot(data_preprocessor.optical_rgb[:, :, 1].flatten(), ax=axis[1])
    sns.distplot(data_preprocessor.optical_rgb[:, :, 2].flatten(), ax=axis[2])
    sns.distplot(data_preprocessor.elevation.flatten(), ax=axis[3])
    sns.distplot(data_preprocessor.slope.flatten(), ax=axis[4])
    plt.tight_layout()
    f.savefig(output_path + "features_distribution.png")
    f.clf()
    plt.close()

    _, axis = plt.subplots(1, 5)
    sns.distplot(data_preprocessor.normalised_optical_rgb[:, :, 0].flatten(), ax=axis[0])
    sns.distplot(data_preprocessor.normalised_optical_rgb[:, :, 1].flatten(), ax=axis[1])
    sns.distplot(data_preprocessor.normalised_optical_rgb[:, :, 2].flatten(), ax=axis[2])
    sns.distplot(data_preprocessor.normalised_elevation.flatten(), ax=axis[3])
    sns.distplot(data_preprocessor.normalised_slope.flatten(), ax=axis[4])
    plt.tight_layout()
    f.savefig(output_path + "normalised_features_distribution.png")
    f.clf()
    plt.close()

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
