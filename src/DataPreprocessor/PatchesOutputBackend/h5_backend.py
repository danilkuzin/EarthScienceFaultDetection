import h5py
import numpy as np

from src.DataPreprocessor.PatchesOutputBackend.sampled_backend import SampledBackend


class H5Backend(SampledBackend):
    def save(self, array: np.array, label:int, path: str):
        with h5py.File(path + '.h5', 'w') as hf:
            hf.create_dataset("data", data=array)

    def load(self, path: str):
        with h5py.File(path+'.h5', 'r') as hf:
            return hf['data'][:]
