# based on  DLPlusCrowdsourcing/CatapultData/learning/suppressed_objects/damaged_building/all_object_non_object_data/joint_via_t_before_after_stronger_prior.py
import h5py

import tensorflow as tf
import numpy as np
from math import ceil
import os

from PIL import Image
from tqdm import trange

from src.learning_tf.net import deepnn_framework, cnn_for_mnist
from src.learning_tf.utils_dataset_processing import next_batch_indices_cycled, shuffle_arrays

tf.enable_eager_execution()

np.set_printoptions(precision=4, suppress=True)
is_print_nn_output = True
np.random.seed(100)
tf.set_random_seed(100)

data_dir = "../../data/Data22012019/"
n_classes = 2
n_epoch = 1000
batch_size = 50
n_train = 10000

# [x, gt_labels, predicted_output, predicted_prob, predictions, keep_prob, \
#  cross_entropy, train_step, accuracy, cnn_without_last_layer] = \
#     deepnn_framework()

#saver = tf.train.Saver()

CNN_train_accuracy = np.zeros_like(range(n_epoch), dtype=np.float64)
CNN_train_prediction = np.zeros_like(range(n_train), dtype=np.int)
CNN_predicted_output = np.zeros_like(n_train, dtype=np.float64)


# def load_patch(class_lbl, ind):
#     patch = Image.open(data_dir + 'learn/train/{}/{}.tif'.format(class_lbl, ind))
#     patch.thumbnail((28, 28), Image.ANTIALIAS)
#     patch = patch.convert('L')
#     patch = np.array(patch)
#     patch = patch / 255
#     patch = patch.flatten()
#     patch = np.expand_dims(patch, axis=1)  # add colour
#     return patch

def create_dataset(data_size, batch_size):
    filenames = [data_dir+"TRAIN_FAULT.tfrecords", data_dir+"TRAIN_NONFAULT.tfrecords"]
    dataset = tf.data.TFRecordDataset(filenames)

    return dataset

sess = tf.Session()
# Create an iterator over the dataset
dataset = create_dataset(data_size=100, batch_size=batch_size)
# Initialize the iterator
#sess.run(iterator.initializer)

# Neural Net Input (images, labels)
#X, Y = iterator.get_next()



for epoch in trange(n_epoch):
    print('\tepoch %d...' % epoch)

    for batch_num in trange(2 * int(ceil(n_train / batch_size))):

        if is_print_nn_output and batch_num % 100 == 0:
            #x = X.eval()
            #gt_labels = Y.eval()
            X = dataset.take(batch_size)
            predicted_output, keep_prob, cnn_without_last_layer = cnn_for_mnist(x)
            redicted_probs = tf.nn.softmax(predicted_output)


            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=gt_labels,
                                                                    logits=predicted_output)
            cross_entropy = tf.reduce_mean(cross_entropy)


            #optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


            correct_prediction = tf.equal(tf.argmax(predicted_output, 1), tf.argmax(gt_labels, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            cur_train_accuracy = tf.reduce_mean(correct_prediction)

            predictions = tf.argmax(predicted_output, 1)

            print('\t\tstep %d, training accuracy w.r.t. VB target %g'
                  % (batch_num, cur_train_accuracy))

            tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

