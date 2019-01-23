from __future__ import print_function
import os
from PIL import Image

data_dir = "../../data/"
image_path_raw = "OneSampleImageData/CNN_faults_context_raw.jpg"
image_path_gt = "OneSampleImageData/CNN_faults_context_lines.jpg"

print(os.listdir(data_dir))

def preprocess_tmp_files():
    im_lines = Image.open(data_dir + image_path_gt)
    im_raw = Image.open(data_dir + image_path_raw)

    assert im_lines.size == im_raw.size

    box_shape = (28, 28)

    os.mkdir(data_dir + "OneSampleImageData/train/")
    os.mkdir(data_dir + "OneSampleImageData/train/fault/")
    os.mkdir(data_dir + "OneSampleImageData/train/nonfault/")
    os.mkdir(data_dir + "OneSampleImageData/valid/")
    os.mkdir(data_dir + "OneSampleImageData/valid/fault/")
    os.mkdir(data_dir + "OneSampleImageData/valid/nonfault/")
    os.mkdir(data_dir + "OneSampleImageData/test/")
    os.mkdir(data_dir + "OneSampleImageData/test/test/")


    for i in range(0, im_lines.size[0] // box_shape[0]):
        for j in range(0, im_lines.size[1] // box_shape[1]):
            print("{}_{}".format(i,j))
            region = im_lines.crop((i*box_shape[0], j*box_shape[1], (i+1)*box_shape[0], (j+1)*box_shape[1]))
            region_pix = region.load()
            red = False
            for k in range(0, box_shape[0]):
                for s in range(0, box_shape[1]):
                    cur_pix = region_pix[k, s]
                    if cur_pix[0] > 200 and cur_pix[1] < 50 and cur_pix[2] < 50:
                        red = True
            region_raw = im_raw.crop((i * box_shape[0], j * box_shape[1], (i + 1) * box_shape[0], (j + 1) * box_shape[1]))
            if red:
                if (i % 17 == 0):
                    region_raw.save(
                        data_dir + "OneSampleImageData/test/test/fault_CNN_faults_context_raw_box_{}_{}.jpg".format(
                            i, j),
                        "JPEG")
                elif (i % 5 != 0):
                    region_raw.save(data_dir + "OneSampleImageData/train/fault/CNN_faults_context_raw_box_{}_{}.jpg".format(i,j),
                            "JPEG")
                else:
                    region_raw.save(
                        data_dir + "OneSampleImageData/valid/fault/CNN_faults_context_raw_box_{}_{}.jpg".format(
                            i, j),
                        "JPEG")
            else:
                if (i % 17 == 0):
                    region_raw.save(
                        data_dir + "OneSampleImageData/test/test/nonfault_CNN_faults_context_raw_box_{}_{}.jpg".format(
                            i, j),
                        "JPEG")
                elif (i % 5 != 0):
                    region_raw.save(data_dir + "OneSampleImageData/train/nonfault/CNN_faults_context_raw_box_{}_{}.jpg".format(i, j),
                            "JPEG")
                else:
                    region_raw.save(
                        data_dir + "OneSampleImageData/valid/nonfault/CNN_faults_context_raw_box_{}_{}.jpg".format(
                            i, j),
                        "JPEG")



preprocess_tmp_files()


