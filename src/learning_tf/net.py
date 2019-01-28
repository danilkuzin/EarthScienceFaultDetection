import tensorflow as tf

class Lenet150Grayscale:


    def model(features, labels, mode):
        input_layer = tf.reshape(features, [-1, 150, 150, 1])

        conv1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)

        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

        dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

        logits = tf.layers.dense(inputs=dropout, units=2)

    def save(self):
        pass

    def load(self):
        pass
    
    def train_op(gt_labels):
        with tf.name_scope('predicted_probs'):
            predicted_probs = tf.nn.softmax(logits)

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=gt_labels,
                                                                    logits=logits)
            cross_entropy = tf.reduce_mean(cross_entropy)

        with tf.name_scope('adam_optimizer'):
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(gt_labels, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)


        accuracy = tf.reduce_mean(correct_prediction)