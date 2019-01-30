import logging

from src.DataPreprocessor.data_preprocessor import DataPreprocessor, Backend, DataOutput, Mode

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    loader = DataPreprocessor("../../data/Region 2 - Muga Puruo/", backend=Backend.GDAL, filename_prefix="mpgr", mode=Mode.TEST)
    loader.prepare_all_patches()
