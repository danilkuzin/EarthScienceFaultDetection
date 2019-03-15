from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.DataPreprocessor.PatchesOutputBackend.in_memory_backend import InMemoryBackend
from src.DataPreprocessor.data_preprocessor import DataPreprocessor

trainable = [0, 1, 6]
data_path = "../data"
data_preprocessor_generators = [
    lambda mode: DataPreprocessor(data_dir="{}/Region 1 - Lopukangri/".format(data_path),
                                  data_io_backend=GdalBackend(),
                                  patches_output_backend=InMemoryBackend(),
                                  mode=mode,
                                  seed=1),
    lambda mode: DataPreprocessor(data_dir="{}/Region 2 - Muga Puruo/".format(data_path),
                                  data_io_backend=GdalBackend(),
                                  patches_output_backend=InMemoryBackend(),
                                  mode=mode,
                                  seed=1),
    lambda mode: DataPreprocessor(data_dir="{}/Region 3 - Muggarboibo/".format(data_path),
                                  data_io_backend=GdalBackend(),
                                  patches_output_backend=InMemoryBackend(),
                                  mode=mode,
                                  seed=1),
    lambda mode: DataPreprocessor(data_dir="{}/Region 4 - Austin-Tonopah/".format(data_path),
                                  data_io_backend=GdalBackend(),
                                  patches_output_backend=InMemoryBackend(),
                                  mode=mode,
                                  seed=1),
    lambda mode: DataPreprocessor(data_dir="{}/Region 5 - Las Vegas Nevada/".format(data_path),
                                  data_io_backend=GdalBackend(),
                                  patches_output_backend=InMemoryBackend(),
                                  mode=mode,
                                  seed=1),
    lambda mode: DataPreprocessor(data_dir="{}/Region 6 - Izmir Turkey/".format(data_path),
                                  data_io_backend=GdalBackend(),
                                  patches_output_backend=InMemoryBackend(),
                                  mode=mode,
                                  seed=1),
    lambda mode: DataPreprocessor(data_dir="{}/Region 7 - Nevada train/".format(data_path),
                                  data_io_backend=GdalBackend(),
                                  patches_output_backend=InMemoryBackend(),
                                  mode=mode,
                                  seed=1),
    lambda mode: DataPreprocessor(data_dir="{}/Region 8 - Nevada test/".format(data_path),
                                  data_io_backend=GdalBackend(),
                                  patches_output_backend=InMemoryBackend(),
                                  mode=mode,
                                  seed=1)
]
