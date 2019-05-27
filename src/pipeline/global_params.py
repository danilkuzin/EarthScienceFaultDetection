from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.DataPreprocessor.PatchesOutputBackend.in_memory_backend import InMemoryBackend
from src.DataPreprocessor.data_preprocessor import DataPreprocessor

trainable = [0, 1, 6, 7, 10]
data_path = "../data"
areas = {"Tibet":0, "Nevada":1, "Turkey":2}
data_preprocessor_params = [
    (f"{data_path}/Region 1 - Lopukangri/", 0),
    (f"{data_path}/Region 2 - Muga Puruo/", 0),
    (f"{data_path}/Region 3 - Muggarboibo/", 0),
    (f"{data_path}/Region 4 - Austin-Tonopah/", 1),
    (f"{data_path}/Region 5 - Las Vegas Nevada/", 1),
    (f"{data_path}/Region 6 - Izmir Turkey/", 2),
    (f"{data_path}/Region 7 - Nevada train/", 1),
    (f"{data_path}/Region 8 - Nevada test/", 1),
    (f"{data_path}/Region 8 - 144036/", 0),
    (f"{data_path}/Region 9 - WRS 143038/", 0),
    (f"{data_path}/Region 10 - 141037/", 0)
]

data_preprocessor_generator = lambda mode, reg_id: DataPreprocessor(
    data_dir=data_preprocessor_paths[reg_id],
    data_io_backend=GdalBackend(),
    patches_output_backend=InMemoryBackend(),
    mode=mode,
    seed=1)
