import pathlib
from typing import Tuple, List

import numpy as np
from tqdm import trange, tqdm
import logging
from src.DataPreprocessor.data_preprocessor import DataPreprocessor
import tensorflow as tf

class KerasTrainer:
    def __init__(self, model_generator, ensemble_size: int):
        self.model_generator = model_generator
        self.ensemble_size = ensemble_size
        self.models = []

    @staticmethod
    def loss(logits, labels):
        return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))

    def train(self, steps_per_epoch, epochs, train_generator):

        #dataset = tf.data.Dataset.from_generator(lambda: train_generator,
        #                                         (tf.float32, tf.int32))
        #iter = dataset.make_one_shot_iterator()

        history_loss = []
        for i in range(self.ensemble_size):
            model = self.model_generator()
            optimizer = tf.train.AdamOptimizer(0.00001)
            for epoch in range(epochs):
                logging.warning(f"epoch:{epoch}")
                for step in range(steps_per_epoch):
                    #data = dataset.batch(batch_size=1)
                    images, labels = next(train_generator)
                    images = tf.constant(images)
                    labels = tf.constant(labels)
                    with tf.GradientTape() as tape:
                        logits = model(images)
                        loss_value = KerasTrainer.loss(logits, labels)
                    logging.warning(f"cur loss:{loss_value.numpy()}")
                    history_loss.append(loss_value.numpy())
                    grads = tape.gradient(loss_value, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
            self.models.append(model)

        return history_loss

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
        res_np = np.array(res_arr)
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
                                 stride: int, batch_size: int, channels:List[int]):
        boxes, probs = [], []
        for patch_coords_batch, patch_batch in tqdm(data_preprocessor.sequential_pass_generator(
                patch_size=patch_size, stride=stride, batch_size=batch_size, channels=channels)):
            boxes.extend(patch_coords_batch)
            probs.extend(self.predict_average(patch_batch))
        boxes = np.stack(boxes, axis=0)
        probs = np.stack(probs, axis=0)
        return boxes, probs
