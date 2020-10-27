import sys
import torch
import numpy as np

import matplotlib.pyplot as plt

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.extend(['../../EarthScienceFaultDetection'])

from src.DataPreprocessor.region_dataset import RegionDataset

from src.config import data_path


def mirror_image(input_image, number_rows_to_mirror, number_columns_to_mirror):
    """
    input_image is np.ndarray encoding image [height, width, channels]. 
    mirror_image mirror number_rows_to_mirror on top and bottom and 
    number_columns_to_mirror on the left and right (both for all channels). 
    Returns enlarged array
    """
    mirrored_part_left = np.flip(input_image[:, :number_columns_to_mirror, :], axis=1)
    mirrored_part_right = np.flip(input_image[:, -number_columns_to_mirror:, :], axis=1)
    mirrored_image = np.concatenate((mirrored_part_left, input_image, mirrored_part_right), axis=1)

    mirrored_part_top = np.flip(mirrored_image[:number_rows_to_mirror, :, :], axis=0)
    mirrored_part_bottom = np.flip(mirrored_image[-number_rows_to_mirror:, :, :], axis=0)
    mirrored_image = np.concatenate((mirrored_part_top, mirrored_image, mirrored_part_bottom), axis=0)

    return mirrored_image


def get_outer_box_coordinates(xmin_center, ymin_center, outer_box_size, center_box_size):
    xshift = (outer_box_size[0] - center_box_size[0]) // 2
    xmin_outer = xmin_center - xshift
    xmax_outer = xmin_outer + outer_box_size[0]

    yshift = (outer_box_size[1] - center_box_size[1]) // 2
    ymin_outer = ymin_center - yshift
    ymax_outer = ymin_outer + outer_box_size[1]

    return (xmin_outer, ymin_outer, xmax_outer, ymax_outer)


def convert_box_to_mirror_image_coordinates(input_box, number_rows_mirrored, number_columns_mirrored):
    """
    input_box is tuple (xmin, ymin, xmax, ymax)
    """
    mirrored_image_coordinated = tuple(sum(x) for x in zip(input_box, 
        (number_columns_mirrored, number_rows_mirrored, number_columns_mirrored, number_rows_mirrored)))

    return mirrored_image_coordinated

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# cnn_model = FCNet()

folder = f"{data_path}/results/semisupervised_individual_class"
training_output = torch.load(folder + '/model_epoch_99.pth', map_location=device)
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

data_preprocessor = RegionDataset(6)

input_image = data_preprocessor.get_full_image(
    channel_list=['optical_rgb', 'elevation', 'slope', 'nir',
                  'topographic_roughness'])
im_width = input_image.shape[1]
im_height = input_image.shape[0]

# mirror on edges
image_patch_size = 736
output_patch_size = 736

mirrored_image = mirror_image(input_image, image_patch_size, image_patch_size)

image_patch_coordinate_list = []
center_image_patch_coordinate_list = []
for i in range(0, im_width, output_patch_size):
    xmin_center = i
    if i + output_patch_size < im_width:
        xmax_center = i + output_patch_size
    else:
        xmax_center = im_width

    for j in range(0, im_height, output_patch_size):
        ymin_center = j

        if j + output_patch_size < im_height:
            ymax_center = j + output_patch_size
        else:
            ymax_center = im_height

        center_box = (xmin_center, ymin_center, xmax_center, ymax_center)
        center_image_patch_coordinate_list.append(center_box)

        outer_box = get_outer_box_coordinates(
            xmin_center, ymin_center, (image_patch_size, image_patch_size),
            (output_patch_size, output_patch_size))
        outer_box_in_mirrored_image_coordinates = \
            convert_box_to_mirror_image_coordinates(
                outer_box, image_patch_size, image_patch_size)

        image_patch_coordinate_list.append(
            outer_box_in_mirrored_image_coordinates)

full_prediction = np.zeros((3, input_image.shape[0], input_image.shape[1]),
                           dtype=np.float)
counter = 0
for image_patch_num in range(len(image_patch_coordinate_list)):
    xmin, ymin, xmax, ymax = image_patch_coordinate_list[image_patch_num]
    sliced_input_image = mirrored_image[ymin:ymax, xmin:xmax, :]
    input_data = np.expand_dims(
        sliced_input_image.astype(np.float32).transpose((2, 0, 1)),
        axis=0)

    prediction = cnn_model(torch.tensor(input_data).to(device))
    prediction_np = prediction.detach().cpu().numpy()

    xmin_center, ymin_center, xmax_center, ymax_center = \
        center_image_patch_coordinate_list[image_patch_num]
    full_prediction[:, ymin_center:ymax_center, xmin_center:xmax_center] = \
        prediction_np[0, :, :ymax_center - ymin_center,
        :xmax_center - xmin_center].copy()

    counter += 1
    print(counter)

np.savez(f"{folder}/prediction_on_6",
         prediction=full_prediction)

# sliced_input_image = input_image[4440:4590, 3528:3678, :]
#
# input_data = np.expand_dims(
#     sliced_input_image.astype(np.float32).transpose((2, 0, 1)),
#     axis=0)
#
# cnn_model.eval()
# # model.requires_grad_(False)
#
# prediction = cnn_model(torch.tensor(input_data))
