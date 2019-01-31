import tensorflow as tf
import numpy as np

from src.LearningKeras.net import cnn_for_mnist_adjust_lr_with_softmax

data_dir = "../../data/"

def train():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        data_dir + 'OneSampleImageData/train',
        color_mode='grayscale',
        target_size=(28, 28),
        batch_size=32,
        shuffle=True,
        class_mode='categorical')

    valid_generator = train_datagen.flow_from_directory(
        data_dir + 'OneSampleImageData/valid',
        color_mode='grayscale',
        target_size=(28, 28),
        batch_size=32,
        shuffle=True,
        class_mode='categorical')

    test_generator = train_datagen.flow_from_directory(
        data_dir + 'OneSampleImageData/test',
        color_mode='grayscale',
        target_size=(28, 28),
        batch_size=1,
        class_mode=None)

    model = cnn_for_mnist_adjust_lr_with_softmax()
    model.fit_generator(train_generator,
                        steps_per_epoch=100,
                        epochs=5,
                        validation_data=valid_generator
                        )

    test_generator.reset()
    pred = model.predict_generator(test_generator, verbose=1)

    predicted_class_indices = np.argmax(pred, axis=1)
    labels = (train_generator.class_indices)
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    print(predictions)

if __name__ == "__main__":
    train()
