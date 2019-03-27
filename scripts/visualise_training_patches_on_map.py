import h5py
import numpy as np

from src.DataPreprocessor.data_preprocessor import Mode
from src.pipeline import global_params

dataset = 0
with h5py.File(f'../train_data/regions_{dataset}/data.h5', 'r') as hf:
    lbls = hf['lbls'][:]

with h5py.File(f'../train_data/regions_{dataset}/data_coords.h5', 'r') as hf:
    coords = hf['coords'][:]

coords = coords.astype(np.int)
lbls = lbls.astype(np.int)
data_preprocessor = global_params.data_preprocessor_generators[0](Mode.TRAIN)
im_w, im_h, _ = data_preprocessor.get_data_shape()

mask_0 = np.zeros((im_w, im_h))
mask_1 = np.zeros((im_w, im_h))
for n in range(coords.shape[0]):
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

data_preprocessor.data_io_backend.write_surface(f'../train_data/regions_{dataset}/data_patches_0.tif', mask_0)
data_preprocessor.data_io_backend.write_surface(f'../train_data/regions_{dataset}/data_patches_1.tif', mask_1)




