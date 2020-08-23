import yaml
import numpy
import seaborn
import matplotlib.pyplot as plt
import torch

from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.config import data_path

folder = f"{data_path}/results/test_training_segmentation_unet_on_6_torch_batchnorm_dice"
saved_model_path = folder + '/model.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_output = torch.load(saved_model_path, map_location=device)


all_train_loss = model_output['all_train_loss']
all_train_iou = model_output['all_train_iou']
all_val_loss = model_output['all_val_loss']
all_val_iou = model_output['all_val_iou']

plt.plot(all_train_loss, label='train')
plt.plot(all_val_loss, label='val')
plt.legend()
plt.savefig(folder + '/loss.png')
plt.show()

plt.plot(all_train_iou, label='train')
plt.plot(all_val_iou, label='val')
plt.legend()
plt.savefig(folder + '/mean_iou.png')
plt.show()




