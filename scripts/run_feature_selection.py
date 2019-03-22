from src.pipeline.training import train
import logging

logging.basicConfig(level=logging.DEBUG)

train(
    train_datasets=[6],
    validation_datasets=[7],
    test_datasets=[0, 1],
    class_probabilities="two-class",
    batch_size=10,
    patch_size=(150, 150),
    channels=[0, 1, 2, 3, 4],
    output_path="train_on_6_features_01234_no_additional/",
    steps_per_epoch=50,
    epochs=10
)
