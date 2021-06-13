import copy
import glob
import os
import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.pipeline_torch.dataset import h5_loader, ToTensor, \
    h5_loader_segmentation
from src.config import data_path


def datasets_on_single_files_torch(regions, channels, train_ratio, batch_size):
    BATCH_SIZE = batch_size

    train_paths = []
    valid_paths = []

    train_dataset_size = 0
    valid_dataset_size = 0

    for reg_id in regions:
        reg_path = pathlib.Path(f'../../{data_path}/train_data/regions_{reg_id}_single_files/')
        all_image_paths = np.array([str(path) for path in list(reg_path.glob('*.h5'))])

        image_count = len(all_image_paths)
        permuted_ind = np.random.permutation(image_count)
        permuted_paths = all_image_paths[permuted_ind]
        train_len = int(image_count * train_ratio)

        train_paths.append(permuted_paths[:train_len])
        train_dataset_size += train_len

        valid_paths.append(permuted_paths[train_len:])
        valid_dataset_size += image_count - train_len

    train_path_ds = h5_loader(np.array(train_paths).flatten(),
                              channels=channels,
                              transform=ToTensor())
    train_dataset = DataLoader(train_path_ds, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=2)

    valid_path_ds = h5_loader(np.array(valid_paths).flatten(),
                              channels=channels,
                              transform=ToTensor())
    valid_dataset = DataLoader(valid_path_ds, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=2)

    return train_dataset, train_dataset_size, valid_dataset, valid_dataset_size


def datasets_on_single_files_torch_segmentation(device, regions,
                                                path_prefix, channels,
                                                train_ratio, batch_size,
                                                num_workers,
                                                transform):
    BATCH_SIZE = batch_size

    train_paths = []
    valid_paths = []

    train_dataset_size = 0
    valid_dataset_size = 0

    for reg_id in regions:
        reg_path = pathlib.Path(path_prefix, 'regions_'+str(reg_id)+'_segmentation_mask')
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
        transform=transform)
    train_dataset = DataLoader(train_path_ds, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=num_workers,
                               drop_last=True) # ToDo remove this

    valid_path_ds = h5_loader_segmentation(
        np.array(valid_paths).flatten(),
        channels=channels,
        transform=ToTensor(device))
    valid_dataset = DataLoader(valid_path_ds, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=num_workers,
                               drop_last=True) # ToDo remove this)

    return train_dataset, train_dataset_size, valid_dataset, valid_dataset_size


def train_on_preloaded_single_files_torch(
        model, train_dataset, train_dataset_size,
        valid_dataset, valid_dataset_size,
        folder, epochs, batch_size, optimizer,
        criterion, scheduler):
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_acc_epoch_num = 0

    all_train_loss = []
    all_train_acc = []

    all_val_loss = []
    all_val_acc = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    datasets = {'train': train_dataset, 'val': valid_dataset}
    dataset_sizes = {'train': train_dataset_size, 'val': valid_dataset_size}

    for epoch in range(epochs):  # loop over the dataset multiple times
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for i, data in enumerate(datasets[phase], 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data['image'], data['label']
                inputs = inputs.to(device)
                labels = labels.to(device)
                # labels = torch.unsqueeze(labels, 1)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #_, preds = torch.max(outputs, 1)
                    loss = criterion(outputs,
                                     labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # print statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]
            # epoch_acc = float(epoch_acc.data.to('cpu').numpy())

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #     phase, epoch_loss, epoch_acc))

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            if phase == 'train':
                all_train_loss.append(epoch_loss)
                # all_train_acc.append(epoch_acc)
            else:
                all_val_loss.append(epoch_loss)
                # all_val_acc.append(epoch_acc)

            # # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_acc_epoch_num = epoch
            #     best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val' and ((epoch + 1) % 5 == 0):
                output = {}
                output['model'] = model
                output['optimizer'] = optimizer
                output['scheduler'] = scheduler
                output['best_acc'] = best_acc
                output['best_acc_epoch_num'] = best_acc_epoch_num
                output['all_train_loss'] = all_train_loss
                output['all_train_acc'] = all_train_acc
                output['all_val_loss'] = all_val_loss
                output['all_val_acc'] = all_val_acc

                torch.save(output, folder + f'/model_epoch_{epoch}.pth')

        print()

    print('Finished Training')

    output = {}
    output['model'] = model
    output['optimizer'] = optimizer
    output['scheduler'] = scheduler
    output['best_acc'] = best_acc
    output['best_acc_epoch_num'] = best_acc_epoch_num
    output['all_train_loss'] = all_train_loss
    output['all_train_acc'] = all_train_acc
    output['all_val_loss'] = all_val_loss
    output['all_val_acc'] = all_val_acc

    torch.save(output, folder + '/model.pth')


def get_jaccard(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1).sum(dim=-1) + \
            y_pred.sum(dim=-2).sum(dim=-1).sum(dim=-1)

    return ((intersection + epsilon) / (union - intersection + epsilon)).mean()


def get_jaccard_non_binary(y_true, y_pred):
    epsilon = 1e-15
    y_true_binary = y_true > 0
    y_pred_class = torch.argmax(y_pred, dim=1)
    y_pred_binary = y_pred_class > 0
    intersection = (y_pred_binary * y_true_binary).sum(dim=-2).sum(dim=-1).sum(dim=-1)
    union = y_true_binary.sum(dim=-2).sum(dim=-1).sum(dim=-1) + \
            y_pred_binary.sum(dim=-2).sum(dim=-1).sum(dim=-1)

    return ((intersection + epsilon) / (union - intersection + epsilon)).mean()


def train_on_preloaded_single_files_torch_unet(
        model, train_dataset, train_dataset_size,
        valid_dataset, valid_dataset_size,
        folder, epochs, batch_size, optimizer,
        criterion, scheduler):
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # check existing checkpoints
    existing_checkpoints = glob.glob(os.path.join(folder, 'model_epoch_*'))
    if len(existing_checkpoints) > 0:
        existing_checkpoints = [os.path.split(checkpoint)[1] for checkpoint in existing_checkpoints]
        epoch_nums = np.array([int(checkpoint.split('_')[-1][:-4]) for checkpoint in existing_checkpoints])
        max_checkpoint = existing_checkpoints[np.argmax(epoch_nums)]

        saved_output = torch.load(os.path.join(folder, max_checkpoint), map_location=device)
        model = saved_output['model']
        optimizer = saved_output['optimizer']
        scheduler = saved_output['scheduler']
        start_epoch = np.max(epoch_nums) + 1

        best_model_wts = saved_output['best_model_wts']
        best_iou = saved_output['best_iou']
        best_iou_epoch_num = saved_output['best_iou_epoch_num']

        all_train_loss = saved_output['all_train_loss']
        all_train_iou = saved_output['all_train_iou']

        all_val_loss = saved_output['all_val_loss']
        all_val_iou = saved_output['all_val_iou']
        print(f'Discovered existing checkpoint, resuming training from epoch={start_epoch}')
    else:
        start_epoch = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        best_iou = 0.0
        best_iou_epoch_num = -1

        all_train_loss = []
        all_train_iou = []

        all_val_loss = []
        all_val_iou = []

    datasets = {'train': train_dataset, 'val': valid_dataset}
    dataset_sizes = {'train': train_dataset_size, 'val': valid_dataset_size}

    for epoch in range(start_epoch, epochs):  # loop over the dataset multiple times
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_iou = 0

            for i, data in enumerate(datasets[phase], 0):
                # print(8*'-')
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data['image'], data['label']
                inputs = inputs.to(device)
                labels = labels.to(device)
                # labels = torch.unsqueeze(labels, 1)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
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
                scheduler.step()

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
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val' and ((epoch + 1) % 5 == 0):
                output = {}
                output['model'] = model
                output['optimizer'] = optimizer
                output['scheduler'] = scheduler
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
    model.load_state_dict(best_model_wts)

    output = {}
    output['model'] = model
    output['optimizer'] = optimizer
    output['scheduler'] = scheduler
    output['best_iou'] = best_iou
    output['best_iou_epoch_num'] = best_iou_epoch_num
    output['all_train_loss'] = all_train_loss
    output['all_train_iou'] = all_train_iou
    output['all_val_loss'] = all_val_loss
    output['all_val_iou'] = all_val_iou

    torch.save(output, folder + '/model.pth')

