import torch
import matplotlib.pyplot as plt

import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.extend(['../../EarthScienceFaultDetection'])

from src.config import data_path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# cnn_model = FCNet()

folder = f"{data_path}/results/semisupervised"
training_output = torch.load(folder + '/model.pth', map_location=device)
cnn_model = training_output['model'].to(device)
cnn_model.eval()

all_train_loss = training_output['all_train_loss']
all_train_iou = training_output['all_train_iou']
all_val_loss = training_output['all_val_loss']
all_val_iou = training_output['all_val_iou']

plt.figure()
plt.plot(all_train_loss, label='train')
plt.plot(all_val_loss, label='val')
plt.legend()
plt.savefig(folder + '/loss.png')
plt.clf()

plt.figure()
plt.plot(all_train_iou, label='train')
plt.plot(all_val_iou, label='val')
plt.legend()
plt.savefig(folder + '/mean_iou.png')
plt.clf()