import pathlib

import h5py

from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.DataPreprocessor.PatchesOutputBackend.in_memory_backend import InMemoryBackend
from src.DataPreprocessor.data_preprocessor import DataPreprocessor, Mode

from src.DataPreprocessor.data_normaliser import  AreaNormaliser, RegionNormaliser
from src.DataPreprocessor.raw_data_preprocessor import RawDataPreprocessor
from src.config import data_preprocessor_params, areas, data_path

import numpy as np
import yaml
import io


def normalise_area(area_id):
    area_normaliser = AreaNormaliser(area_id)
    area_normaliser.normalise()

def normalise_region(region_id, area_ind):
    data_normaliser = RegionNormaliser(region_id, area_ind)
    data_normaliser.load()
    data_normaliser.normalise()
    data_normaliser.save_results()

def preprocess_single_region(path: pathlib.Path, region_id:int):
    raw_data_preprocessor = RawDataPreprocessor(data_dir=path, data_io_backend=GdalBackend())
    raw_data_preprocessor.load()
    output_path = f"{data_path}/preprocessed/{region_id}"
    pathlib.Path(output_path).mkdir(exist_ok=True, parents=True)
    raw_data_preprocessor.write_data(output_path)


for region_id, region_params in enumerate(data_preprocessor_params):
    path = pathlib.Path(region_params[0])
    preprocess_single_region(path=path, region_id=region_id)

for key_area, val_area in areas.items():
    normalise_area(val_area)

for region_id, region_params in enumerate(data_preprocessor_params):
    path = pathlib.Path(region_params[0])
    normalise_region(region_id, area_ind=region_params[1])

