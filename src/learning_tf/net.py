# based on  DLPlusCrowdsourcing/NNArchitectures/two_images_simpler_deepnn_3_classes.py

import tensorflow as tf

def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def cnn_for_mnist(x):
    """cnn_for_mnist builds the graph for a deep net for classifying digits.
    Args:
        x: an input tensor with the dimensions (N_examples, 784), where 784 is the
        number of pixels in a standard MNIST image.
        Returns:
            A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
            equal to the logits of classifying the digit into one of 10 classes (the
            digits 0-9). keep_prob is a scalar placeholder for the probability of
            dropout.
    """
    # Reshape to use within a convolutional neural net.
    #  Last dimension is for "features" - there is only one here, since images are
    #  grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First layer
    # First convolutional layer - maps one grayscale image to 64 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # First pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second layer
    # Second convolutional layer -- maps 64 feature maps to 128.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Third layer
    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x128 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 3 classes
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 2])
        b_fc2 = bias_variable([2])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob, h_fc1_drop




def deepnn_framework():
    x = tf.placeholder(tf.float32, shape=[None, 784, 1])
    gt_labels = tf.placeholder(tf.float32, shape=[None, 2])

    predicted_output, keep_prob, cnn_without_last_layer = cnn_for_mnist(x)

    with tf.name_scope('predicted_probs'):
        predicted_probs = tf.nn.softmax(predicted_output)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=gt_labels,
                                                                logits=predicted_output)
        cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(predicted_output, 1), tf.argmax(gt_labels, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    # tf.Print(predicted_prob, [predicted_prob], message='predicted_prob')
    # tf.Print(gt_labels, [gt_labels], message='labels')

    with tf.name_scope('predictions'):
        predictions = tf.argmax(predicted_output, 1)

    return [x, gt_labels, predicted_output, predicted_probs, predictions, keep_prob, cross_entropy, train_step,
            accuracy, cnn_without_last_layer]



