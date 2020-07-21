import pathlib
from typing import Tuple, List
import tensorflow as tf

import numpy as np
import torch
from tqdm import trange, tqdm

from src.DataPreprocessor.region_dataset import RegionDataset


class TorchTrainer:
    def __init__(self, model=None):
        self.model = model

    # def train(self, steps_per_epoch, epochs, train_generator, valid_generator):
    #     history = self.model.fit_generator(generator=train_generator,
    #                                       steps_per_epoch=steps_per_epoch,
    #                                       epochs=epochs,
    #                                       validation_data=valid_generator,
    #                                       validation_steps=5,
    #                                       workers=0,
    #                                       use_multiprocessing=False)
    #     return history
    #
    # def train_array(self, x_train, y_train, x_val, y_val):
    #     callbacks = [
    #         tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=32, write_graph=True,
    #                                        write_grads=True, write_images=True),
    #         tf.keras.callbacks.CSVLogger(filename='log.csv', separator=',', append=False)
    #     ]
    #
    #     history = self.model.fit(x=x_train, y=y_train, batch_size=10, epochs=5, verbose=2, callbacks=callbacks,
    #                             validation_data=(x_val, y_val), validation_freq=1)
    #
    #     return history
    #
    # def evaluate(self):
    #     pass
    #
    # def save(self, output_path):
    #     pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    #     self.model.save_weights(output_path + '/model.h5')
    #
    def load(self, input_path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_output = torch.load(f'{input_path}/model.pth',
                                  map_location=device)
        self.model = model_output['model']
        self.model.eval()

    #
    # def apply_for_test(self, data_generator, num_test_samples):
    #     true_labels = []
    #     predicted_labels = []
    #
    #     for _ in trange(num_test_samples):
    #         images, lbls = next(data_generator)
    #         for i in range(images.shape[0]):
    #             probs = self.model.predict(images[i])
    #             true_labels.append(lbls[i])
    #             predicted_labels.append(np.argmax(probs))
    #
    #     true_labels = np.array(true_labels)
    #     predicted_labels = np.array(predicted_labels)
    #
    #     return np.mean(true_labels == predicted_labels)

    def apply_for_sliding_window(self, data_preprocessor: RegionDataset, patch_size: Tuple[int, int],
                                 stride: int, batch_size: int, channels:List[int]):
        self.model.eval()
        boxes, probs = [], []
        for patch_coords_batch, patch_batch in tqdm(data_preprocessor.sequential_pass_generator(
                patch_size=patch_size, stride=stride, batch_size=batch_size, channels=channels)):
            boxes.extend(patch_coords_batch)
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            patch_batch = patch_batch.transpose((0, 3, 1, 2))
            patch_batch_tensor = torch.from_numpy(
                patch_batch.astype(np.float32))
            probs_tensor = self.model(patch_batch_tensor)
            probs.extend(probs_tensor.detach().numpy())
        boxes = np.stack(boxes, axis=0)
        probs = np.stack(probs, axis=0)
        return boxes, probs
