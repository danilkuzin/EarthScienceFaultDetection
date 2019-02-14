from PIL import Image
from tqdm import trange
import numpy as np

from src.DataPreprocessor.PatchesOutputBackend.sampled_backend import SampledBackend


class FolderBackend(SampledBackend):
    def save(self, array: np.array, label: int, path: str):
        for i in trange(array.shape[0]):
            patch_im = Image.fromarray(array[i])
            patch_im.save(path + "/{}.tif".format(i))
