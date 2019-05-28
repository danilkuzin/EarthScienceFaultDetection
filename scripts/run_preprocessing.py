import pathlib

import h5py

from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.DataPreprocessor.PatchesOutputBackend.in_memory_backend import InMemoryBackend
from src.DataPreprocessor.data_preprocessor import DataPreprocessor, Mode
from src.config import data_preprocessor_params, areas, data_path

import numpy as np
import yaml
import io

def normalise_area(area_ind):
    associated_regions = []
    for region_id, region in enumerate(data_preprocessor_params):
        if region[1] == area_ind:
            associated_regions.append(region_id)

    print(f"area: {area_ind}, associated regions: {associated_regions}")
    elevations_list = []
    for region_id in associated_regions:
        preprocessed_data_path = f"{data_path}/preprocessed/{region_id}/data.h5"
        with h5py.File(preprocessed_data_path, 'r') as hf:
            elevations_list.append(hf["elevation"][:])

    for i in range(len(elevations_list)):
        elevations_list[i] = elevations_list[i].flatten()

    elevations_array = np.concatenate(elevations_list)
    mean_elevation = np.mean(elevations_array)
    std_elevation = np.std(elevations_array)

    output_path = f"{data_path}/normalised"
    pathlib.Path(output_path).mkdir(exist_ok=True, parents=True)

    with io.open(f"{output_path}/area_{area_ind}.yaml", 'w', encoding='utf8') as outfile:
        yaml.dump({"mean_elevation": float(mean_elevation), "std_elevation": float(std_elevation)}, outfile, default_flow_style=False, allow_unicode=True)


def normalise_region(region_id, area_ind):
    preprocessed_data_path = f"{data_path}/preprocessed/{region_id}/data.h5"
    with h5py.File(preprocessed_data_path, 'r') as hf:
        optical_r = hf["optical_r"][:]
        optical_g = hf["optical_g"][:]
        optical_b = hf["optical_b"][:]
        elevation = hf["elevation"][:]
        slope = hf["slope"][:]

    def normalise(feature):
        feature_mean, feature_std = np.mean(feature), np.std(feature)
        feature_normalised = (feature - feature_mean) / feature_std
        return feature_mean, feature_std, feature_normalised

    # optical_r_mean, optical_r_std, optical_r_normalised = normalise(optical_r)
    # optical_g_mean, optical_g_std, optical_g_normalised = normalise(optical_g)
    # optical_b_mean, optical_b_std, optical_b_normalised = normalise(optical_b)
    # slope_mean, slope_std, slope_normalised = normalise(slope)

    optical_mean = 127.5
    optical_std = 255.
    optical_r_normalised = (optical_r - optical_mean) / optical_std
    optical_g_normalised = (optical_g - optical_mean) / optical_std
    optical_b_normalised = (optical_b - optical_mean) / optical_std

    input_path = f"{data_path}/normalised"
    with open(f"{input_path}/area_{area_ind}.yaml", 'r') as stream:
        features_areawide = yaml.safe_load(stream)

    elevation_mean, elevation_std = features_areawide['mean_elevation'], features_areawide['std_elevation']
    elevation_normalised = (elevation - elevation_mean) / elevation_std

    slope_mean = 45.
    slope_std = 90.
    slope_normalised = (slope - slope_mean) / slope_std

    with h5py.File(f"{input_path}/{region_id}_data.h5", 'w') as hf:
        dict(elevation=None, slope=None, optical_rgb=None, nir=None, ultrablue=None, swir1=None, swir2=None,
             panchromatic=None, curve=None, erosion=None)
        hf.create_dataset("elevation", data=elevation_normalised)
        hf.create_dataset("slope", data=slope_normalised)
        hf.create_dataset("optical_r", data=optical_r_normalised)
        hf.create_dataset("optical_g", data=optical_g_normalised)
        hf.create_dataset("optical_b", data=optical_b_normalised)


def preprocess_single_region(path: pathlib.Path, area_id: int, trainable:bool, region_id:int):
    #todo move trainable inside instead of mode? or remove it and decide from data inside
    if trainable:
        data_preprocessor = DataPreprocessor(
        data_dir=str(path)+"/",
        data_io_backend=GdalBackend(),
        patches_output_backend=InMemoryBackend(),
        mode=Mode.TRAIN,
        seed=1)
    else:
        data_preprocessor = DataPreprocessor(
        data_dir=str(path)+"/",
        data_io_backend=GdalBackend(),
        patches_output_backend=InMemoryBackend(),
        mode=Mode.TEST,
        seed=1)


    output_path = f"{data_path}/preprocessed/{region_id}"
    pathlib.Path(output_path).mkdir(exist_ok=True, parents=True)
    data_preprocessor.write_data(output_path)


for region_id, region_params in enumerate(data_preprocessor_params):
    path = pathlib.Path(region_params[0])
    preprocess_single_region(path=path, area_id=region_params[1], trainable=region_params[2], region_id=region_id)

for key_area, val_area in areas.items():
    normalise_area(val_area)

for region_id, region_params in enumerate(data_preprocessor_params):
    path = pathlib.Path(region_params[0])
    normalise_region(region_id, area_ind=region_params[1])

