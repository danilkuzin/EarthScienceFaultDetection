import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

sys.path.extend(
    [
        os.path.join('..', '..', 'EarthScienceFaultDetection'),
        os.path.join('..', '..', 'EarthScienceFaultDetection', 'hrnet'),
    ]
)
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

from hrnet.lib.config import config, update_config
from hrnet.lib.models.seg_hrnet import get_seg_model

from src.pipeline_torch.dataset import ToTensor, RandomRotation, ToTFImageInput, \
    RandomHorizontalFlip, RandomVerticalFlip, RandomBrightness, RandomContrast, \
    FromTFImageOutput

from src.LearningTorch.net_architecture import UNet, LossBinary, FocalLoss, \
    LossCrossDice, LossMulti, LossMultiSemiSupervised, Res34_Unet, \
    LossMultiSemiSupervisedEachClass
from src.pipeline_torch.training import datasets_on_single_files_torch, \
    train_on_preloaded_single_files_torch_unet, \
    datasets_on_single_files_torch_segmentation

from src.config import data_path

import torchvision


np.random.seed(1000)

device = torch.device("cuda:0") # if torch.cuda.is_available() else "cpu")

# cnn_model = Res34_Unet(n_input_channels=2, n_classes=3)

n_input_channels = 1
n_classes = 3


class Configs:
    cfg = None
    opts = None


args = Configs

args.cfg = os.path.join(
    '..',
    'hrnet',
    'experiments',
    'cityscapes',
    'seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml'
)
args.opts = ['DATASET.NUM_CLASSES', n_classes]


update_config(config, args)

# config.defrost()
# config.DATASET.NUM_CLASSES = n_classes
# config.freeze()

cnn_model = get_seg_model(config)
cnn_model.conv1 = torch.nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)

criterion = LossMultiSemiSupervisedEachClass(
    device=device, nll_weight=1, jaccard_weight=5,
    focal_weight=12, ignore_classes_for_nll=[3],
    ignore_classes_for_jaccard=[0],
    alpha=0.9, gamma=2, reduction='mean')

#LossMultiSemiSupervised(jaccard_weight=5, ignore_class_for_nll=3, ignore_classes_for_jaccard=[0])
# LossMulti(jaccard_weight=5, num_classes=3) # nn.CrossEntropyLoss()
# LossCrossDice(jaccard_weight=5)
# FocalLoss(reduction='mean')
# FocalLoss(gamma=0.5, alpha=0.5)
# LossBinary(jaccard_weight=5) # nn.BCEWithLogitsLoss()

optimizer = optim.Adam(cnn_model.parameters(), lr=1e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10,
                                       gamma=0.1)

# im = np.random.randint(255, size=(1, 5, 156, 156)).astype(np.float32)
# im_tensor = torch.tensor(im)
#
# output = cnn_model(im_tensor)
# print(output.shape)


batch_size = 4
num_workers = 0

cnn_model = cnn_model.to(device)

transform = torchvision.transforms.Compose([
    ToTFImageInput(),
    RandomRotation(),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomBrightness(),
    RandomContrast(),
    FromTFImageOutput(),
    ToTensor(device),
    ]
)

train_dataset, train_dataset_size, valid_dataset, valid_dataset_size = \
    datasets_on_single_files_torch_segmentation(
        device=device,
        regions=[6], path_prefix=os.path.join(data_path, 'train_data'),
        channels=[0],
        train_ratio=0.8, batch_size=batch_size,
        num_workers=num_workers,
        transform=transform
)

train_on_preloaded_single_files_torch_unet(
    cnn_model, train_dataset, train_dataset_size, valid_dataset,
    valid_dataset_size,
    folder=os.path.join(data_path, 'results', 'nevada_hazmap_hrnet_elev'),
    epochs=100,
    batch_size=batch_size,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=exp_lr_scheduler)

