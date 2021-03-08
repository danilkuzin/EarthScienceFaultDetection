import pathlib

from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.DataPreprocessor.area_normaliser import AreaNormaliser
from src.DataPreprocessor.raw_data_preprocessor import RawDataPreprocessor
from src.DataPreprocessor.region_normaliser import RegionNormaliser
from src.config import data_preprocessor_params, areas


def normalise_area(area_id, landsat: bool = False):
    area_normaliser = AreaNormaliser(area_id)
    area_normaliser.normalise(landsat)


def normalise_region(region_id, area_ind, is_roughness_log=False):
    data_normaliser = RegionNormaliser(region_id, area_ind)
    data_normaliser.load()
    data_normaliser.normalise(is_roughness_log)
    data_normaliser.save_results()


def preprocess_single_region(path: pathlib.Path, region_id: int,
                             landsat: bool = False):
    raw_data_preprocessor = RawDataPreprocessor(
        data_dir=path, data_io_backend=GdalBackend(), region_id=region_id)
    if landsat:
        raw_data_preprocessor.load_landsat()
        raw_data_preprocessor.write_data_landsat()
    else:
        raw_data_preprocessor.load()
        raw_data_preprocessor.write_data()


# for region_id, region_params in enumerate(data_preprocessor_params):
#     path = pathlib.Path(region_params[0])
#     preprocess_single_region(path=path, region_id=region_id)
# preprocess_single_region(
#     path=pathlib.Path(data_preprocessor_params[12][0]),
#     region_id=12,
#     landsat=True
# )

# for key_area, val_area in areas.items():
#     normalise_area(val_area)
# normalise_area(3, landsat=True)
#
# for region_id, region_params in enumerate(data_preprocessor_params):
#     path = pathlib.Path(region_params[0])
#     normalise_region(region_id, area_ind=region_params[1])

normalise_region(12, area_ind=data_preprocessor_params[12][1],
                 is_roughness_log=False)
