import copy
import pathlib
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.extend(['../../EarthScienceFaultDetection'])
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

from src.pipeline_torch.dataset_torchvision import h5_loader_segmentation
from src.pipeline_torch.transforms_torchvision import RandomRotation, \
    RandomHorizontalFlip, RandomVerticalFlip, ColorJitter

from src.LearningTorch.net_architecture import Res34_Unet, \
    LossMultiSemiSupervisedEachClass
#from src.pipeline_torch.training import get_jaccard_non_binary

def get_jaccard_non_binary(y_true, y_pred):
    epsilon = 1e-15
    y_true_binary = y_true > 0
    y_pred_class = torch.argmax(y_pred, dim=1)
    y_pred_binary = y_pred_class > 0
    intersection = (y_pred_binary * y_true_binary).sum(dim=-2).sum(dim=-1).sum(dim=-1)
    union = y_true_binary.sum(dim=-2).sum(dim=-1).sum(dim=-1) + \
                   y_pred_binary.sum(dim=-2).sum(dim=-1).sum(dim=-1)

    return ((intersection + epsilon) / (union - intersection + epsilon)).mean()

data_path = '/mnt/data/datasets/DataForEarthScienceFaultDetection'

# np.random.seed(1000)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cnn_model = Res34_Unet(n_input_channels=7, n_classes=3)
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

optimizer = torch.optim.Adam(cnn_model.parameters(), lr=1e-4)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                       gamma=0.1)

# im = np.random.randint(255, size=(1, 5, 156, 156)).astype(np.float32)
# im_tensor = torch.tensor(im)
#
# output = cnn_model(im_tensor)
# print(output.shape)


batch_size = 4
num_workers = 0

cnn_model = cnn_model.to(device)

transform_train = torch.nn.Sequential(
    # ToTensor(),
    RandomRotation(degrees=[-45., 45.]),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    ColorJitter(brightness=10, contrast=10),
)

# train_dataset, train_dataset_size, valid_dataset, valid_dataset_size = \
#     datasets_on_single_files_torch_segmentation(
#         device=device,
#         regions=[6], path_prefix=f'{data_path}/train_data',
#         channels=[0, 1, 2, 3, 4, 5, 6],
#         train_ratio=0.8, batch_size=batch_size,
#         num_workers=num_workers,
#         transform=transform_train
# )
###
BATCH_SIZE = batch_size

train_paths = []
valid_paths = []

train_dataset_size = 0
valid_dataset_size = 0

regions = [6]
path_prefix = f'{data_path}/train_data'
train_ratio = 0.8
channels=[0, 1, 2, 3, 4, 5, 6]

for reg_id in regions:
    reg_path = pathlib.Path(path_prefix + f'/regions_{reg_id}_segmentation_mask/')
    all_image_paths = np.array([str(path) for path in list(reg_path.glob('*.h5'))])

    image_count = len(all_image_paths)
    permuted_ind = np.random.permutation(image_count)
    permuted_paths = all_image_paths[permuted_ind]
    train_len = int(image_count * train_ratio)

    train_paths.append(permuted_paths[:train_len])
    train_dataset_size += train_len

    valid_paths.append(permuted_paths[train_len:])
    valid_dataset_size += image_count - train_len

train_path_ds = h5_loader_segmentation(
    np.array(train_paths).flatten(),
    channels=channels,
    device=device,
    transform=transform_train)
train_dataset = DataLoader(train_path_ds, batch_size=BATCH_SIZE,
                           shuffle=True, num_workers=num_workers,
                           drop_last=True) # ToDo remove this

valid_path_ds = h5_loader_segmentation(
    np.array(valid_paths).flatten(),
    channels=channels,
    device=device
)
valid_dataset = DataLoader(valid_path_ds, batch_size=BATCH_SIZE,
                           shuffle=True, num_workers=num_workers,
                           drop_last=True) # ToDo remove this)

###
# train_on_preloaded_single_files_torch_unet(
#     cnn_model, train_dataset, train_dataset_size, valid_dataset, valid_dataset_size,
#     folder=f"{data_path}/results/semisupervised_one_class",
#     epochs=100,
#     batch_size=batch_size,
#     optimizer=optimizer,
#     criterion=criterion,
#     scheduler=exp_lr_scheduler)
folder=f"{data_path}/results/semisupervised_one_class_upd_transforms"
pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
epochs=100

best_model_wts = copy.deepcopy(cnn_model.state_dict())
best_iou = 0.0
best_iou_epoch_num = -1

all_train_loss = []
all_train_iou = []

all_val_loss = []
all_val_iou = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

datasets = {'train': train_dataset, 'val': valid_dataset}
dataset_sizes = {'train': train_dataset_size, 'val': valid_dataset_size}

for epoch in range(epochs):  # loop over the dataset multiple times
    print('Epoch {}/{}'.format(epoch, epochs - 1))
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            cnn_model.train()  # Set model to training mode
        else:
            cnn_model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_iou = 0

        for i, data in enumerate(datasets[phase], 0):
            # print(8*'-')
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data['image'], data['label']
            # inputs = inputs.to(device)
            # labels = labels.to(device)
            # labels = torch.unsqueeze(labels, 1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = cnn_model(inputs)
                #_, preds = torch.max(outputs, 1)
                # dim1_cropping = (labels.shape[1] - outputs.shape[-2]) // 2
                # dim2_cropping = (labels.shape[2] - outputs.shape[-1]) // 2
                # labels = labels[:, dim1_cropping:-dim1_cropping,
                #          dim2_cropping:-dim2_cropping]
                labels = labels.long()
                loss = criterion(outputs,
                                 labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # print statistics
            running_loss += loss.item() * inputs.size(0)
            running_iou += get_jaccard_non_binary(
                labels, outputs).item()
            # running_iou += get_jaccard(labels,
            #                            (outputs > 0).float()).item()

        if phase == 'train':
            exp_lr_scheduler.step()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_iou = running_iou / dataset_sizes[phase]
        # epoch_iou = float(epoch_acc.data.to('cpu').numpy())

        print('{} Loss: {:.4f} IoU: {:.4f}'.format(
            phase, epoch_loss, epoch_iou))

        # print('{} Loss: {:.4f}'.format(
        #     phase, epoch_loss))

        if phase == 'train':
            all_train_loss.append(epoch_loss)
            all_train_iou.append(epoch_iou)
        else:
            all_val_loss.append(epoch_loss)
            all_val_iou.append(epoch_iou)

        # # deep copy the model
        if phase == 'val' and epoch_iou > best_iou:
            best_iou = epoch_iou
            best_iou_epoch_num = epoch
            best_model_wts = copy.deepcopy(cnn_model.state_dict())

        if phase == 'val' and ((epoch + 1) % 5 == 0):
            output = {}
            output['model'] = cnn_model
            output['optimizer'] = optimizer
            output['scheduler'] = exp_lr_scheduler
            output['best_iou'] = best_iou
            output['best_iou_epoch_num'] = best_iou_epoch_num
            output['best_model_wts'] = best_model_wts
            output['all_train_loss'] = all_train_loss
            output['all_train_iou'] = all_train_iou
            output['all_val_loss'] = all_val_loss
            output['all_val_iou'] = all_val_iou

            torch.save(output, folder + f'/model_epoch_{epoch}.pth')

    print()

print('Finished Training')

print(f'Best val IoU: {best_iou} achieved at epoch No: {best_iou_epoch_num}')

# load best model weights
cnn_model.load_state_dict(best_model_wts)

output = {}
output['model'] = cnn_model
output['optimizer'] = optimizer
output['scheduler'] = exp_lr_scheduler
output['best_iou'] = best_iou
output['best_iou_epoch_num'] = best_iou_epoch_num
output['all_train_loss'] = all_train_loss
output['all_train_iou'] = all_train_iou
output['all_val_loss'] = all_val_loss
output['all_val_iou'] = all_val_iou

torch.save(output, folder + '/model.pth')

