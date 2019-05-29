import sys
sys.path.extend(['../../EarthScienceFaultDetection'])
import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

tf.set_random_seed(5)
np.random.seed(5)

from src.pipeline.generate_data import generate_data, generate_data_single_files

#generate_data(datasets=[6], size=2000)
generate_data_single_files(dataset_inds=[6], size=15000, lookalike_ratio=[None, None, None])
