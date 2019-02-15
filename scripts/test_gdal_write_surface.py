from src.DataPreprocessor.DataIOBackend.gdal_backend import GdalBackend
from src.DataPreprocessor.PatchesOutputBackend.in_memory_backend import InMemoryBackend
from src.DataPreprocessor.data_preprocessor import DataPreprocessor, Mode

data = DataPreprocessor(data_dir="../data/Region 1 - Lopukangri/",
                     data_io_backend=GdalBackend(),
                     patches_output_backend=InMemoryBackend(),
                     filename_prefix="tibet",
                     mode=Mode.TRAIN,
                     seed=1)

import numpy as np
img = np.random.randint(0, 1000, (500, 500))

data.data_io_backend.write_surface("gdal_test.vrt", img)