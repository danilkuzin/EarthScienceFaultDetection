from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.DataPreprocessor.PatchesOutputBackend.in_memory_backend import InMemoryBackend
from src.DataPreprocessor.data_preprocessor import DataPreprocessor, Mode

data_preprocessor_generators_train = [
    lambda: DataPreprocessor(data_dir="../../data/Region 1 - Lopukangri/",
                     data_io_backend=GdalBackend(),
                     patches_output_backend=InMemoryBackend(),
                     filename_prefix="tibet",
                     mode=Mode.TRAIN,
                     seed=1),
    lambda: DataPreprocessor(data_dir="../../data/Region 2 - Muga Puruo/",
                     data_io_backend=GdalBackend(),
                     patches_output_backend=InMemoryBackend(),
                     filename_prefix="mpgr",
                     mode=Mode.TRAIN,
                     seed=1),
    lambda: DataPreprocessor(data_dir="../../data/Region 3 - Muggarboibo/",
                     data_io_backend=GdalBackend(),
                     patches_output_backend=InMemoryBackend(),
                     filename_prefix="gyrc1",
                     mode=Mode.TEST,
                     seed=1)
]

data_preprocessor_generators_test = [
    lambda: DataPreprocessor(data_dir="../data/Region 1 - Lopukangri/",
                     data_io_backend=GdalBackend(),
                     patches_output_backend=InMemoryBackend(),
                     filename_prefix="tibet",
                     mode=Mode.TEST,
                     seed=1),
    lambda: DataPreprocessor(data_dir="../data/Region 2 - Muga Puruo/",
                     data_io_backend=GdalBackend(),
                     patches_output_backend=InMemoryBackend(),
                     filename_prefix="mpgr",
                     mode=Mode.TEST,
                     seed=1),
    lambda: DataPreprocessor(data_dir="../data/Region 3 - Muggarboibo/",
                     data_io_backend=GdalBackend(),
                     patches_output_backend=InMemoryBackend(),
                     filename_prefix="gyrc1",
                     mode=Mode.TEST,
                     seed=1)
]