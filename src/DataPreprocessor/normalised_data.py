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
            if self.channels['slope'] is not None:
                hf.create_dataset("slope", data=self.channels['slope'])
            hf.create_dataset("optical_r", data=self.channels['optical_rgb'][:, :, 0])
            hf.create_dataset("optical_g", data=self.channels['optical_rgb'][:, :, 1])
            hf.create_dataset("optical_b", data=self.channels['optical_rgb'][:, :, 2])
            if 'nir' in self.channels:
                hf.create_dataset("nir", data=self.channels['nir'])
            if 'topographic_roughness' in self.channels:
                hf.create_dataset("topographic_roughness",
                                  data=self.channels['topographic_roughness'])
            if 'flow' in self.channels:
                hf.create_dataset("flow",
                                  data=self.channels['flow'])
            if 'ultrablue' in self.channels:
                hf.create_dataset("ultrablue",
                                  data=self.channels['ultrablue'])
            if 'swir1' in self.channels:
                hf.create_dataset("swir1",
                                  data=self.channels['swir1'])
            if 'swir2' in self.channels:
                hf.create_dataset("swir2",
                                  data=self.channels['swir2'])
            hf.create_dataset("features", data=self.features)

    def load(self):
        with h5py.File(f"{self.storage_path}/data.h5", 'r') as hf:
            optical_r = hf["optical_r"][:]
            optical_g = hf["optical_g"][:]
            optical_b = hf["optical_b"][:]
            self.channels['optical_rgb'] = np.stack((optical_r, optical_g, optical_b), axis=-1)
            self.channels['elevation'] = hf["elevation"][:]
            if "slope" in hf:
                self.channels['slope'] = hf["slope"][:]
            else:
                self.channels['slope'] = None
            if "nir" in hf:
                self.channels['nir'] = hf["nir"][:]
            else:
                self.channels['nir'] = None
            if "topographic_roughness" in hf:
                self.channels['topographic_roughness'] = \
                    hf["topographic_roughness"][:]
            else:
                self.channels['topographic_roughness'] = None
            if "ultrablue" in hf:
                self.channels['ultrablue'] = hf["ultrablue"][:]
            else:
                self.channels['ultrablue'] = None
            if "swir1" in hf:
                self.channels['swir1'] = hf["swir1"][:]
            else:
                self.channels['swir1'] = None
            if "swir2" in hf:
                self.channels['swir2'] = hf["swir2"][:]
            else:
                self.channels['swir2'] = None
            if "flow" in hf:
                self.channels['flow'] = hf["flow"][:]
            else:
                self.channels['flow'] = None
            self.features = hf["features"][:]


