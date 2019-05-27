import pathlib

from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.DataPreprocessor.PatchesOutputBackend.in_memory_backend import InMemoryBackend
from src.DataPreprocessor.data_preprocessor import DataPreprocessor, Mode
from src.config import data_preprocessor_params, areas, data_path


# def process_area(area_ind, area):
#     associated_regions = list(filter(lambda x: x[1] == area_ind, data_preprocessor_params))
#     print(f"area: {area}, associated regions: {associated_regions}")
#     for region in associated_regions:
#
#
#
# for area_ind, area in enumerate(areas):
#     process_area(area_ind, area)

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
