import h5py

from src.config import data_path
import io
import yaml
import pathlib
import numpy as np

# persistence handling for preprocessed data
class PreprocessedData:
    def __init__(self, region_id):
        self.channels = dict(
            elevation=None, slope=None, optical_rgb=None, nir=None,
            ultrablue=None, swir1=None, swir2=None, panchromatic=None,
            curve=None, erosion=None, topographic_roughness=None,
            flow=None,
        )
        self.features = None
        self.storage_path = f"{data_path}/preprocessed/{region_id}"

    def save(self):
        pathlib.Path(self.storage_path).mkdir(exist_ok=True, parents=True)
        with h5py.File(f"{self.storage_path}/data.h5", 'w') as hf:
            dict(elevation=None, slope=None, optical_rgb=None, nir=None, ultrablue=None, swir1=None, swir2=None,
                 panchromatic=None, curve=None, erosion=None)
            hf.create_dataset("elevation", data=self.channels['elevation'])
            hf.create_dataset("slope", data=self.channels['slope'])
            hf.create_dataset("optical_r", data=self.channels['optical_rgb'][:,:,0])
            hf.create_dataset("optical_g", data=self.channels['optical_rgb'][:,:,1])
            hf.create_dataset("optical_b", data=self.channels['optical_rgb'][:,:,2])
            hf.create_dataset("nir", data=self.channels['nir'])
            hf.create_dataset("ultrablue", data=self.channels['ultrablue'])
            hf.create_dataset("swir1", data=self.channels['swir1'])
            hf.create_dataset("swir2", data=self.channels['swir2'])
            hf.create_dataset("panchromatic", data=self.channels['panchromatic'])
            hf.create_dataset("curve", data=self.channels['curve'])
            hf.create_dataset("erosion", data=self.channels['erosion'])
            hf.create_dataset("topographic_roughness", data=self.channels['topographic_roughness'])
            hf.create_dataset("features", data=self.features)

    def save_landsat(self):
        pathlib.Path(self.storage_path).mkdir(exist_ok=True, parents=True)
        with h5py.File(f"{self.storage_path}/data.h5", 'w') as hf:
            # dict(elevation=None, optical_rgb=None, nir=None,
            #      swir1=None, swir2=None,
            #      topographic_roughness=None, flow=None)
            hf.create_dataset("elevation", data=self.channels['elevation'])
            if self.channels['slope'] is not None:
                hf.create_dataset("slope", data=self.channels['slope'])
            hf.create_dataset("optical_r", data=self.channels['optical_rgb'][:,:,0])
            hf.create_dataset("optical_g", data=self.channels['optical_rgb'][:,:,1])
            hf.create_dataset("optical_b", data=self.channels['optical_rgb'][:,:,2])
            hf.create_dataset("nir", data=self.channels['nir'])
            hf.create_dataset("swir1", data=self.channels['swir1'])
            hf.create_dataset("swir2", data=self.channels['swir2'])
            hf.create_dataset("topographic_roughness", data=self.channels['topographic_roughness'])
            hf.create_dataset("flow",
                              data=self.channels['flow'])
            hf.create_dataset("erosion",
                              data=self.channels['erosion'])
            hf.create_dataset("sar1",
                              data=self.channels['sar1'])
            hf.create_dataset("sar2",
                              data=self.channels['sar2'])
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
            if "panchromatic" in hf:
                self.channels['panchromatic'] = hf["panchromatic"][:]
            else:
                self.channels['panchromatic'] = None
            if "curve" in hf:
                self.channels['curve'] = hf["curve"][:]
            else:
                self.channels['curve'] = None
            if "erosion" in hf:
                self.channels['erosion'] = hf["erosion"][:]
            else:
                self.channels['erosion'] = None
            if "flow" in hf:
                self.channels['flow'] = hf['flow'][:]
            else:
                self.channels['flow'] = None
            if "sar1" in hf:
                self.channels['sar1'] = hf['sar1'][:]
            else:
                self.channels['sar1'] = None
            if "sar2" in hf:
                self.channels['sar2'] = hf['sar2'][:]
            else:
                self.channels['sar2'] = None
            self.features = hf["features"][:]


    def load_landsat(self):
        with h5py.File(f"{self.storage_path}/data.h5", 'r') as hf:
            optical_r = hf["optical_r"][:]
            optical_g = hf["optical_g"][:]
            optical_b = hf["optical_b"][:]
            self.channels['optical_rgb'] = np.stack((optical_r, optical_g, optical_b), axis=-1)
            self.channels['elevation'] = hf["elevation"][:]
            self.channels['erosion'] = hf["erosion"][:]
            self.channels['nir'] = hf["nir"][:]
            self.channels['topographic_roughness'] = hf["topographic_roughness"][:]
            self.channels['flow'] = hf["flow"][:]
            self.channels['swir1'] = hf["swir1"][:]
            self.channels['swir2'] = hf["swir2"][:]
            self.channels['sar1'] = hf["sar1"][:]
            self.channels['sar2'] = hf["sar2"][:]
            self.features = hf["features"][:]
