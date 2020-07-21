import copy
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

def datasets_on_single_files_torch_segmentation(regions, channels, train_ratio, batch_size):
    BATCH_SIZE = batch_size

    train_paths = []
    valid_paths = []

    train_dataset_size = 0
    valid_dataset_size = 0

    for reg_id in regions:
        reg_path = pathlib.Path(f'{data_path}/train_data/regions_{reg_id}_segmentation_mask/')
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
        transform=ToTensor())
    train_dataset = DataLoader(train_path_ds, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=2)

    valid_path_ds = h5_loader_segmentation(
        np.array(valid_paths).flatten(),
        channels=channels,
        transform=ToTensor())
    valid_dataset = DataLoader(valid_path_ds, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=2)

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
