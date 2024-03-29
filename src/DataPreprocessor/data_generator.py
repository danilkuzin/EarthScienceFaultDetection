from src.DataPreprocessor.region_dataset import RegionDataset
from typing import List, Tuple
import numpy as np


class DataGenerator:
    """
    A collection of data_preprocessors that feed data into training pipeline
    """
    def __init__(self, preprocessors: List[RegionDataset]):
        self.preprocessors = preprocessors

    def generator_3class(self, batch_size: int, class_probabilities: np.array, patch_size: Tuple[int, int], channels: np.array, verbose: int):
        while True:
            img_batches = []
            lbl_batches = []
            coords_batches = []
            for preprocessor in self.preprocessors:
                img_batch, lbl_batch, coords_batch = next(preprocessor.train_generator_3class(
                    batch_size=batch_size,
                    class_probabilities=class_probabilities,
                    patch_size=patch_size,
                    channels=channels,
                    verbose=verbose))
                img_batches.append(img_batch)
                lbl_batches.append(lbl_batch)
                coords_batches.append(coords_batch)
            yield img_batches, lbl_batches

    def generator_2class_lookalikes_with_nonfaults(self, batch_size: int, class_probabilities: np.array, patch_size: Tuple[int, int], channels: np.array, verbose: int):
        while True:
            img_batches = []
            lbl_batches = []
            coords_batches = []
            for preprocessor in self.preprocessors:
                img_batch, lbl_batch, coords_batch = next(preprocessor.train_generator_2class_lookalikes_with_nonfaults(
                    batch_size=batch_size,
                    class_probabilities=class_probabilities,
                    patch_size=patch_size,
                    channels=channels,
                    verbose=verbose))
                img_batches.append(img_batch)
                lbl_batches.append(lbl_batch)
                coords_batches.append(coords_batch)
            yield np.concatenate(img_batches, axis=0), np.concatenate(lbl_batches, axis=0), np.concatenate(coords_batches, axis=0)

    def create_datasets(self, class_probabilities: str, patch_size: Tuple[int, int], channels: List[int], size: int, verbose=0):

        if class_probabilities == "equal":
            class_probabilities_int = np.array([1. / 3, 1. / 3, 1. / 3])
            joint_generator = self.generator_3class(
                batch_size=size,
                class_probabilities=class_probabilities_int,
                patch_size=patch_size,
                channels=np.array(channels),
                verbose=verbose)

        elif class_probabilities == "two-class":
            class_probabilities_int = np.array([0.5, 0.25, 0.25])
            joint_generator = self.generator_2class_lookalikes_with_nonfaults(
                batch_size=size,
                class_probabilities=class_probabilities_int,
                patch_size=patch_size,
                channels=np.array(channels),
                verbose=verbose)

        else:
            class_probabilities_int = np.array(class_probabilities)
            joint_generator = self.generator_2class_lookalikes_with_nonfaults(
                batch_size=size,
                class_probabilities=class_probabilities_int,
                patch_size=patch_size,
                channels=np.array(channels),
                verbose=verbose)

        imgs, lbls, coords = next(joint_generator)
        return imgs, lbls, coords



