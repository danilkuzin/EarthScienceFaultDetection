# based on  DLPlusCrowdsourcing/CatapultData/learning/suppressed_objects/damaged_building/all_object_non_object_data/joint_via_t_before_after_stronger_prior.py
import h5py

import tensorflow as tf
import numpy as np
from math import ceil
import os

from PIL import Image
from tqdm import trange

from src.learning_tf.net import deepnn_framework
from src.learning_tf.utils_dataset_processing import next_batch_indices_cycled, shuffle_arrays

np.set_printoptions(precision=4, suppress=True)
is_print_nn_output = True
np.random.seed(100)
tf.set_random_seed(100)

data_dir = "../../data/Data22012019/"
n_classes = 2
n_epoch = 1000
batch_size = 50
n_train = 10000

[x, gt_labels, predicted_output, predicted_prob, predictions, keep_prob, \
 cross_entropy, train_step, accuracy, cnn_without_last_layer] = \
    deepnn_framework()

saver = tf.train.Saver()

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

def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    image_feature_description = {
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'depth': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.VarLenFeature(tf.float32),
    }

    return tf.parse_single_example(example_proto, image_feature_description)

def create_dataset(data_size, batch_size):
    filenames = [data_dir+"TRAIN_FAULT.tfrecord", data_dir+"TRAIN_NONFAULT.tfrecord"]
    dataset = tf.data.TFRecordDataset(filenames).shuffle(buffer_size=10).repeat()
    dataset = dataset.map(_parse_image_function)
    return dataset

sess = tf.Session()
# Create an iterator over the dataset
dataset = create_dataset(data_size=100, batch_size=batch_size)
iterator = dataset.make_initializable_iterator()
# Initialize the iterator
sess.run(iterator.initializer)

# Neural Net Input (images, labels)
X = iterator.get_next()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer, feed_dict={})

    for epoch in trange(n_epoch):
        print('\tepoch %d...' % epoch)

        for batch_num in trange(2 * int(ceil(n_train / batch_size))):
            images = np.zeros((batch_size, 150, 150, 5))
            labels = np.zeros((batch_size, 2))
            for ind in range(batch_size):
                images[ind] = X['image_raw'].eval().values.reshape(150, 150, 5)
                lbl = X['label'].eval()
                if lbl == 1:
                    labels[ind, 0] = 1
                elif lbl ==3:
                    labels[ind, 1] = 1

            if is_print_nn_output and batch_num % 100 == 0:

                cur_train_accuracy = accuracy.eval(feed_dict={
                    x: images,
                    gt_labels: labels,
                    keep_prob: 1.0})
                print('\t\tstep %d, training accuracy w.r.t. VB target %g'
                      % (batch_num, cur_train_accuracy))

            train_step.run(feed_dict={x: images, gt_labels: labels, keep_prob: 0.5})

        # cnn output prior softmax for numerically stable computations
        # for i in range(0, images.shape[0], batch_size):
        #     if i + batch_size > images.shape[0]:
        #         nn_output = predicted_output.eval(feed_dict={
        #             x: images[i:], keep_prob: 1.0})
        #         CNN_predicted_output[i:] = nn_output
        #     else:
        #         nn_output = predicted_output.eval(feed_dict={
        #             x: images[i:i + batch_size], keep_prob: 1.0})
        #         CNN_predicted_output[i:i + batch_size] = nn_output
        #
        # CNN_train_prediction = np.argmax(CNN_predicted_output, 1)
        #
        # CNN_train_accuracy[epoch] = np.mean(CNN_train_prediction == np.argmax(ibcc_labels, axis=1))
        # print('\tepoch %d, train accuracy w.r.t. pure IBCC labels %g'
        #       % (epoch, CNN_train_accuracy[epoch]))

    # saver.save(sess, path_upwards + results_path + experiment_name +
    #            "trained_models/joint_via_t_before_after_stonger_prior.ckpt")

# np.savez(path_upwards + results_path + experiment_name + 'joint_via_t_results_before_after_imagery_stronger_prior',
#          CNN_train_accuracy=CNN_train_accuracy,
#          CNN_train_prediction=CNN_train_prediction)
