from src.pipeline.training import train
import tensorflow as tf

import logging

#logging.basicConfig(level=logging.DEBUG)

#tf.enable_eager_execution()
train(
    train_datasets=[6],
    class_probabilities="two-class",
    batch_size=10,
    patch_size=(150, 150),
    channels=[0, 1, 2, 3, 4],
    ensemble_size=1,
    output_path="train_on_6_only_longer/",
    steps_per_epoch=10,
    epochs=50
)
