import h5py

from src.config import data_path
import io
import yaml
import pathlib
import numpy as np

class NormalisedData:
    def __init__(self, region_id):
        self.channels = dict(elevation=None, slope=None, optical_rgb=None)
        self.features = None
        self.storage_path = f"{data_path}/normalised/{region_id}"
        self.region_id = region_id

    def save(self):
        pathlib.Path(self.storage_path).mkdir(exist_ok=True, parents=True)
        with h5py.File(f"{self.storage_path}/data.h5", 'w') as hf:
            hf.create_dataset("elevation", data=self.channels['elevation'])
            hf.create_dataset("slope", data=self.channels['slope'])
            hf.create_dataset("optical_r", data=self.channels['optical_rgb'][:, :, 0])
            hf.create_dataset("optical_g", data=self.channels['optical_rgb'][:, :, 1])
            hf.create_dataset("optical_b", data=self.channels['optical_rgb'][:, :, 2])
            hf.create_dataset("features", data=self.features)

    def load(self):
        with h5py.File(f"{self.storage_path}/data.h5", 'r') as hf:
            optical_r = hf["optical_r"][:]
            optical_g = hf["optical_g"][:]
            optical_b = hf["optical_b"][:]
            self.channels['optical_rgb'] = np.stack((optical_r, optical_g, optical_b), axis=-1)
            self.channels['elevation'] = hf["elevation"][:]
            self.channels['slope'] = hf["slope"][:]
            self.features = hf["features"][:]

