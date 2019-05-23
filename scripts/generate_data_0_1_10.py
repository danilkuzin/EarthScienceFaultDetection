import sys
sys.path.extend(['../../EarthScienceFaultDetection'])
import tensorflow as tf

tf.enable_eager_execution()

from src.pipeline.generate_data import generate_data

generate_data(datasets=[0, 1, 10], size=10000)
