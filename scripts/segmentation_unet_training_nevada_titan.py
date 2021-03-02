import sys
import numpy as np
import torch


import torch.optim as optim
from torch.optim import lr_scheduler

sys.path.extend(['../../EarthScienceFaultDetection'])
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cnn_model = Res34_Unet(n_input_channels=8, n_classes=3)
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
        regions=[6], path_prefix=f'{data_path}/train_data',
        channels=[0, 1, 2, 3, 4, 5, 6, 7],
        train_ratio=0.8, batch_size=batch_size,
        num_workers=num_workers,
        transform=transform
)

train_on_preloaded_single_files_torch_unet(
    cnn_model, train_dataset, train_dataset_size, valid_dataset,
    valid_dataset_size,
    folder=f"{data_path}/results/nevada_hazmap_semisupervised",
    epochs=100,
    batch_size=batch_size,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=exp_lr_scheduler)

