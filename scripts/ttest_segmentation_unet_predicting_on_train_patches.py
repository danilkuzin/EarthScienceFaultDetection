import pathlib
import sys

import h5py
import torch
import numpy as np

import os

from tqdm import tqdm

import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from src.DataPreprocessor.region_dataset import RegionDataset
from src.pipeline_torch.predicting import predict_torch

sys.path.extend(['../../EarthScienceFaultDetection'])

from src.LearningTorch.net_architecture import FCNet
from src.pipeline.predicting import predict, postprocess
from src.config import data_path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# cnn_model = FCNet()

folder = f"{data_path}/results/test_training_segmentation_unet_on_6_torch_augmentation"
training_output = torch.load(folder + '/model.pth', map_location=device)
cnn_model = training_output['model']
cnn_model.eval()

# predict_torch(datasets=[6], models_folder=f"{data_path}/results/test_training_on_6_torch", classes=2,
#         channels=[0, 1, 2, 3, 4], stride=50, batch_size=16)
# postprocess(datasets=[0], models_folder=f"{data_path}/results/test_training_on_6_torch", heatmap_mode="mean")

reg_id = 6
path_prefix = f'{data_path}/train_data'
reg_path = pathlib.Path(path_prefix + f'/regions_{reg_id}_segmentation_mask/')
all_image_paths = np.array([str(path) for path in list(reg_path.glob('*.h5'))])

channels = [0, 1, 2, 3, 4]

for image_path in tqdm(all_image_paths):
    with h5py.File(image_path, 'r') as hf:
        img = hf['img'][:].astype(np.float32)[:, :, channels]
        lbl = hf['lbl'][:].astype(np.float32)
        coord = hf['coord'][:].astype(np.int32)

    img = img.transpose((2, 0, 1))

    image = torch.from_numpy(img).to(device)
    label = torch.from_numpy(lbl).to(device)

    prediction = cnn_model(torch.tensor(image))


    plt.imshow(img)
    plt.title('image')
    plt.show()
    plt.imshow(lbl)
    plt.title('true')
    plt.show()
    plt.imshow(prediction)
    plt.title('pred')
    plt.show()





