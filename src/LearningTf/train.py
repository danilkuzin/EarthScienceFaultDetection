from src.LearningKeras import net_architecture
from src.LearningTf import net_layers
import tensorflow as tf
import numpy as np

from src.pipeline import global_params


def cnn_150x150x1_3class_fn(features, labels, mode):
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

    pool2_flat = tf.reshape(pool2, [-1, 37 * 37 * 64])

    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense, rate=0.5)
    #dropout = tf.layers.dropout(
    #    inputs=dense, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=3)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def train_eager(data_generator):
    dataset = tf.data.Dataset.from_generator(generator=data_generator, output_types=(tf.int64, tf.int64))
    model = net_architecture.cnn_150x150x5()
    optimizer = tf.train.MomentumOptimizer(flags_obj.lr, flags_obj.momentum)
    for (batch, (images, labels)) in enumerate(dataset):
        with tf.GradientTape() as tape:
            logits = model(images)
            loss_value = loss(logits, labels)
            #tf.contrib.summary.scalar('loss', loss_value)
            #tf.contrib.summary.scalar('accuracy', compute_accuracy(logits, labels))
        grads = tape.gradient(loss_value, model.variables)
        optimizer.apply_gradients(
            zip(grads, model.variables), global_step=step_counter)
        if log_interval and batch % log_interval == 0:
            rate = log_interval / (time.time() - start)
            print('Step #%d\tLoss: %.6f (%d steps/sec)' % (batch, loss_value, rate))
    start = time.time()


def train_lazy(data_generator):
    def input_func_gen():
        shapes = ((150, 150, 1), (1))
        dataset = tf.data.Dataset.from_generator(generator=data_generator,
                                                 output_types=(tf.float32, tf.int32))
        dataset = dataset.batch(4)
        # dataset = dataset.repeat(20)
        iterator = dataset.make_one_shot_iterator()
        features_tensors, labels = iterator.get_next()
        #features = {'x': features_tensors}
        return features_tensors, labels

    classifier = tf.estimator.Estimator(
        model_fn=cnn_150x150x1_3class_fn, model_dir="tf/logs")
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    # summary_hook = tf.train.SummarySaverHook(
    #     1,
    #     output_dir='tmp/tf',
    #     summary_op=tf.summary.merge_all())
    # Train the model
    # train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": train_data},
    #     y=train_labels,
    #     batch_size=100,
    #     num_epochs=None,
    #     shuffle=True)
    # train one step and display the probabilties
    for i in range(10):
        classifier.train(
            input_fn=input_func_gen,
            steps=1,
            hooks=[logging_hook])

if __name__ == "__main__":
    data_preprocessor_gen = global_params.data_preprocessor_generators_train[0]
    data_preprocessor = data_preprocessor_gen()
    train_lazy(lambda: data_preprocessor.train_generator_3class(batch_size=4, class_probabilities=np.array([1./3, 1./3, 1./3]), patch_size=(150, 150), channels=np.array([3])))
