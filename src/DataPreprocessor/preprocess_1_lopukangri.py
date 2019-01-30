import logging

from src.DataPreprocessor.data_preprocessor import DataPreprocessor, Backend, DataOutput, Mode

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    loader = DataPreprocessor("../../data/Region 1 - Lopukangri/", backend=Backend.GDAL, filename_prefix="tibet", mode=Mode.TRAIN)
    loader.prepare_datasets(output=DataOutput.TFRECORD)
    loader.prepare_all_patches()
