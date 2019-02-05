import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf

from src.DataPreprocessor.data_preprocessor import DataPreprocessor, Mode
from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.DataPreprocessor.data_visualiser import DataVisualiser

np.random.seed(1)
tf.set_random_seed(2)

dataiobackend = GdalBackend()
data_preprocessor = DataPreprocessor(data_dir="data/Region 1 - Lopukangri/",
                              backend=dataiobackend,
                              filename_prefix="tibet",
                              mode=Mode.TRAIN,
                              seed=1)

data_visualiser = DataVisualiser(data_preprocessor)

plt.imshow(data_visualiser.get_optical_rgb_with_features_mask(opacity=90))

f, (ax1, ax2, ax3) = plt.subplots(1, 3)
sns.distplot(data_preprocessor.optical_rgb[:, :, 0].flatten(), ax=ax1) # r channel
sns.distplot(data_preprocessor.optical_rgb[:, :, 1].flatten(), ax=ax2) # g channel
sns.distplot(data_preprocessor.optical_rgb[:, :, 2].flatten(), ax=ax3) # b channel

plt.imshow(data_preprocessor.elevation)
plt.colorbar()

plt.imshow(data_visualiser.get_elevation_with_features_mask(opacity=90))

sns.distplot(data_preprocessor.elevation.flatten())

plt.imshow(data_preprocessor.slope)
plt.colorbar()

plt.imshow(data_visualiser.get_slope_with_features_mask(opacity=90))

sns.distplot(data_preprocessor.slope.flatten())

f, (ax1, ax2, ax3) = plt.subplots(1, 3)
sns.distplot(data_preprocessor.normalised_optical_rgb[:, :, 0].flatten(), ax=ax1) # r channel
sns.distplot(data_preprocessor.normalised_optical_rgb[:, :, 1].flatten(), ax=ax2) # g channel
sns.distplot(data_preprocessor.normalised_optical_rgb[:, :, 2].flatten(), ax=ax3) # b channel
plt.tight_layout()

sns.distplot(data_preprocessor.normalised_elevation.flatten())

sns.distplot(data_preprocessor.normalised_slope.flatten())

num_patches = 7
patch_size = (150, 150)
bands = 5

patches = np.zeros((num_patches, patch_size[0], patch_size[1], bands))
for i in range(num_patches):
    patches[i] = data_preprocessor.sample_fault_patch(patch_size=patch_size)

for i in range(num_patches):
    cur_patch = patches[i]
    rgb, elevation, slope = data_preprocessor.denormalise(cur_patch)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(rgb)
    ax2.imshow(elevation)
    ax3.imshow(slope)
    plt.tight_layout()
plt.show()

patches = np.zeros((num_patches, patch_size[0], patch_size[1], bands))
for i in range(num_patches):
    patches[i] = data_preprocessor.sample_fault_lookalike_patch(patch_size)

for i in range(num_patches):
    cur_patch = patches[i]
    rgb, elevation, slope = data_preprocessor.denormalise(cur_patch)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(rgb)
    ax2.imshow(elevation)
    ax3.imshow(slope)
    plt.tight_layout()
plt.show()

patches = np.zeros((num_patches, patch_size[0], patch_size[1], bands))
for i in range(num_patches):
    patches[i] = data_preprocessor.sample_nonfault_patch(patch_size)

for i in range(num_patches):
    cur_patch = patches[i]
    rgb, elevation, slope = data_preprocessor.denormalise(cur_patch)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(rgb)
    ax2.imshow(elevation)
    ax3.imshow(slope)
    plt.tight_layout()
plt.show()
