from src.DataPreprocessor.PatchesOutputBackend.backend import PatchesOutputBackend
import numpy as np
import h5py


class H5Backend(PatchesOutputBackend):
    def save(self, array: np.array, label:int, path: str):
        with h5py.File(path + '.h5', 'w') as hf:
            hf.create_dataset("data", data=array)

    def load(self, path: str):
        with h5py.File(path+'.h5', 'r') as hf:
            return hf['data'][:]
