import tensorflow as tf
import numpy as np

from src.DataPreprocessor.preprocess_data_22012019 import DataPreprocessor22012019
from src.nn.net import cnn_for_mnist_adjust_lr_with_softmax

data_dir = "../../data/Data22012019/"

def train():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        data_dir + 'learn/train',
        color_mode='grayscale',
        target_size=(28, 28),
        batch_size=32,
        shuffle=True,
        class_mode='categorical')

    valid_generator = train_datagen.flow_from_directory(
        data_dir + 'learn/valid',
        color_mode='grayscale',
        target_size=(28, 28),
        batch_size=32,
        shuffle=True,
        class_mode='categorical')

    test_generator = train_datagen.flow_from_directory(
        data_dir + 'learn/test',
        color_mode='grayscale',
        target_size=(28, 28),
        batch_size=1,
        class_mode=None)

    model = cnn_for_mnist_adjust_lr_with_softmax()
    model.fit_generator(train_generator,
                        steps_per_epoch=1000,
                        epochs=15,
                        validation_data=valid_generator
                        )

    test_generator.reset()
    pred = model.predict_generator(test_generator, verbose=1)

    predicted_class_indices = np.argmax(pred, axis=1)
    labels = (train_generator.class_indices)
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    print(predictions)

def train_with_gen():
    generator_state = DataPreprocessor22012019("../../data/Data22012019/")

    dataset = tf.data.Dataset.from_generator(
        lambda: generator_state,
        (tf.float32, tf.float32),
        (tf.TensorShape([None, 28, 28, 1]), tf.TensorShape([None, 2]))
    )

    model = cnn_for_mnist_adjust_lr_with_softmax()
    model.fit(dataset.map(lambda x, y: tf.image.resize_images).make_one_shot_iterator(),
              steps_per_epoch=10,
              epochs=5,
              verbose=1)



if __name__ == "__main__":
    train_with_gen()
