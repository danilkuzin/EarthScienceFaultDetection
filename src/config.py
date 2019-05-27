from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.DataPreprocessor.PatchesOutputBackend.in_memory_backend import InMemoryBackend
from src.DataPreprocessor.data_preprocessor import DataPreprocessor

trainable = [0, 1, 6, 7, 10]
data_path = "../../DataForEarthScienceFaultDetection"
areas = {"Tibet":0, "Nevada":1, "Turkey":2}
# path, area, trainable
data_preprocessor_params = [
    (f"{data_path}/raw_data/Region 1 - Lopukangri/", 0, True),
    (f"{data_path}/raw_data/Region 2 - Muga Puruo/", 0, True),
    (f"{data_path}/raw_data/Region 3 - Muggarboibo/", 0, False),
    (f"{data_path}/raw_data/Region 4 - Austin-Tonopah/", 1, False),
    (f"{data_path}/raw_data/Region 5 - Las Vegas Nevada/", 1, False),
    (f"{data_path}/raw_data/Region 6 - Izmir Turkey/", 2, False),
    (f"{data_path}/raw_data/Region 7 - Nevada train/", 1, True),
    (f"{data_path}/raw_data/Region 8 - Nevada test/", 1, True),
    (f"{data_path}/raw_data/Region 8 - 144036/", 0, False),
    (f"{data_path}/raw_data/Region 9 - WRS 143038/", 0, False),
    (f"{data_path}/raw_data/Region 10 - 141037/", 0, True)
]


