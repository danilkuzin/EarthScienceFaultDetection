import pathlib
import h5py

from src.DataPreprocessor.data_generator import DataGenerator
from src.DataPreprocessor.data_preprocessor import Mode
from src.pipeline import global_params

datasets = [1]

preprocessors = [global_params.data_preprocessor_generators[ind](Mode.TRAIN) for ind in datasets]
data_generator = DataGenerator(preprocessors=preprocessors)

imgs, lbls = data_generator.create_datasets(
    class_probabilities="two-class",
    patch_size=(150, 150),
    channels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    size=1000)

output_path = "../train_data/regions_{}/".format("".join([str(i) for i in datasets]))

pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

with h5py.File(output_path + '/data.h5', 'w') as hf:
    hf.create_dataset("imgs", data=imgs)
    hf.create_dataset("lbls", data=lbls)
