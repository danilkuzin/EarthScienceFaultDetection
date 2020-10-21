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

folder = f"{data_path}/results/semisupervised"
training_output = torch.load(folder + '/model.pth', map_location=device)
cnn_model = training_output['model'].to(device)
cnn_model.eval()

# predict_torch(datasets=[6], models_folder=f"{data_path}/results/test_training_on_6_torch", classes=2,
#         channels=[0, 1, 2, 3, 4], stride=50, batch_size=16)
# postprocess(datasets=[0], models_folder=f"{data_path}/results/test_training_on_6_torch", heatmap_mode="mean")

reg_id = 6
path_prefix = f'{data_path}/train_data'
reg_path = pathlib.Path(path_prefix + f'/regions_{reg_id}_segmentation_mask/')
all_image_paths = np.array([str(path) for path in list(reg_path.glob('*.h5'))])

channels = [0, 1, 2, 3, 4, 5, 6]

os.makedirs(f"{folder}/prediction_on_train_patches/", exist_ok=True)

for i in range(20):
    image_path = all_image_paths[i]
    with h5py.File(image_path, 'r') as hf:
        img = hf['img'][:].astype(np.float32)[:, :, channels]
        lbl = hf['lbl'][:].astype(np.float32)
        coord = hf['coord'][:].astype(np.int32)

    image = img.transpose((2, 0, 1))

    image = np.expand_dims(image, axis=0)

    image = torch.from_numpy(image).to(device)
    label = torch.from_numpy(lbl).to(device)

    prediction = cnn_model(image)

    fig, axs = plt.subplots(1, 5)
    axs[0].imshow(img[:, :, 3])
    axs[0].set_title('elevation')
    axs[0].axis('off')
    axs[1].imshow(lbl == 1)
    axs[1].set_title('true mask front range')
    axs[1].axis('off')
    axs[2].imshow(prediction.cpu().detach().numpy()[1])
    axs[2].set_title('prediction front range')
    axs[2].axis('off')
    axs[3].imshow(lbl == 2)
    axs[3].set_title('true mask basin')
    axs[3].axis('off')
    axs[4].imshow(prediction.cpu().detach().numpy()[2])
    axs[4].set_title('prediction basin')
    axs[4].axis('off')
    plt.savefig(f"{folder}/prediction_on_train_patches/patch_{i}.png")
    plt.cla()





