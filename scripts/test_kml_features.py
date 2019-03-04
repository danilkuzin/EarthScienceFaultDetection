from src.DataPreprocessor.data_preprocessor import Mode
from src.pipeline.global_params import data_preprocessor_generators
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib

data_preprocessor = data_preprocessor_generators[0](mode=Mode.TRAIN)
gt = data_preprocessor.data_io_backend.gdal_options['geotransform']

lookalike_lines = []
for l_ind in range(1, 3):
    with open("{}/additional_data/Look_Alike_1_{}.kml.utm".format(data_preprocessor.data_dir, l_ind)) as f:
        content = f.readlines()

    coords = []
    for line in content:
        rr = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line)
        if len(rr) == 3:
            coords.append(list(map(float, rr)))

    pixel_coords = []

    for coord in coords:
        Xpixel = int((coord[0] - gt[0]) / gt[1])
        Ypixel = int((coord[1] - gt[3]) / gt[5])
        pixel_coords.append([Xpixel, Ypixel])

    lookalike_lines.append(np.array(pixel_coords))

nonfault_lines = []
for l_ind in range(1, 4):
    with open("{}/additional_data/Not_Fault_1_{}.kml.utm".format(data_preprocessor.data_dir, l_ind)) as f:
        content = f.readlines()

    coords = []
    for line in content:
        rr = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line)
        if len(rr) == 3:
            coords.append(list(map(float, rr)))

    pixel_coords = []

    for coord in coords:
        Xpixel = int((coord[0] - gt[0]) / gt[1])
        Ypixel = int((coord[1] - gt[3]) / gt[5])
        pixel_coords.append([Xpixel, Ypixel])

    nonfault_lines.append(np.array(pixel_coords))

plt.imshow(data_preprocessor.channels['optical_rgb'])
ax = plt.gca()

lookalike_patches=[]
for lookalike_line in lookalike_lines:
    polygon = Polygon(lookalike_line, True)
    lookalike_patches.append(polygon)
p = PatchCollection(lookalike_patches, color='g', alpha=0.4)
ax.add_collection(p)

nonfault_patches=[]
for nonfault_line in nonfault_lines:
    polygon = Polygon(nonfault_line, True)
    nonfault_patches.append(polygon)
p = PatchCollection(nonfault_patches, color='b', alpha=0.4)
ax.add_collection(p)

# for lookalike_line in lookalike_lines:
#     plt.plot(lookalike_line[:, 0], lookalike_line[:, 1], marker='o', color='green')
# for nonfault_line in nonfault_lines:
#     plt.plot(nonfault_line[:, 0], nonfault_line[:, 1], marker='o', color='red')

plt.savefig('example.png')


