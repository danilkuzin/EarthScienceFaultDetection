from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.DataPreprocessor.PatchesOutputBackend.in_memory_backend import InMemoryBackend
from src.DataPreprocessor.data_preprocessor import DataPreprocessor

trainable = [0, 1, 6, 7, 10]
data_path = "../data"
data_preprocessor_paths = [
    f"{data_path}/Region 1 - Lopukangri/",
    f"{data_path}/Region 2 - Muga Puruo/",
    f"{data_path}/Region 3 - Muggarboibo/",
    f"{data_path}/Region 4 - Austin-Tonopah/",
    f"{data_path}/Region 5 - Las Vegas Nevada/",
    f"{data_path}/Region 6 - Izmir Turkey/",
    f"{data_path}/Region 7 - Nevada train/",
    f"{data_path}/Region 8 - Nevada test/",
    f"{data_path}/Region 8 - 144036/",
    f"{data_path}/Region 9 - WRS 143038/",
    f"{data_path}/Region 10 - 141037/"
]

data_preprocessor_generator = lambda mode, reg_id: DataPreprocessor(
    data_dir=data_preprocessor_paths[reg_id],
    data_io_backend=GdalBackend(),
    patches_output_backend=InMemoryBackend(),
    mode=mode,
    seed=1)
