import pathlib
from typing import Tuple

import numpy as np
from tqdm import trange

from src.DataPreprocessor.data_preprocessor import DataPreprocessor


class KerasTrainer:
    def __init__(self, model_generator, ensemble_size: int):
        self.model_generator = model_generator
        self.ensemble_size = ensemble_size
        self.models = []

    def train(self, steps_per_epoch, epochs, train_generator):
        history_arr = []
        for i in range(self.ensemble_size):
            model = self.model_generator()
            history = model.fit_generator(train_generator,
                                          steps_per_epoch=steps_per_epoch,
                                          epochs=epochs,
                                          validation_data=train_generator,
                                          validation_steps=5,
                                          workers=0,
                                          use_multiprocessing=False)

            self.models.append(model)
            history_arr.append(history)
        return history_arr

    def save(self, output_path):
        for i in range(self.ensemble_size):
            pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
            self.models[i].save_weights(output_path + '/model_{}.h5'.format(i))

    def load(self, input_path):
        for i in range(self.ensemble_size):
            model = self.model_generator()
            model.load_weights('{}/model_{}.h5'.format(input_path, i))
            self.models.append(model)

    def predict_average(self, patch):
        res_arr = []
        for model in self.models:
            res_arr.append(model.predict(patch))
        res_np = np.concatenate(res_arr)
        res_avg = np.mean(res_np, axis=0)
        return res_avg

    def apply_for_test(self, data_generator, num_test_samples):
        true_labels = []
        predicted_labels = []

        for _ in trange(num_test_samples):
            images, lbls = next(data_generator)
            for i in range(images.shape[0]):
                probs = self.predict_average(images[i])
                true_labels.append(lbls[i])
                predicted_labels.append(np.argmax(probs))

        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)

        return np.mean(true_labels == predicted_labels)

    def apply_for_sliding_window(self, data_preprocessor: DataPreprocessor, patch_size: Tuple[int, int],
                                 stride: int, batch_size: int):
        boxes, probs = [], []
        for patch_coords_batch, patch_batch in data_preprocessor.sequential_pass_generator(
                patch_size=patch_size, stride=stride, batch_size=batch_size):
            boxes.append(patch_coords_batch)
            probs.append(self.predict_average(patch_batch))
        boxes = np.concatenate(boxes, axis=0)
        probs = np.concatenate(probs, axis=0)
        return boxes, probs
