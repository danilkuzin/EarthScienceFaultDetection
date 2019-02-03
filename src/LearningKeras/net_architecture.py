import tensorflow as tf


def cnn_28x28x1(lr=1e-4):
    cnn_model = tf.keras.models.Sequential()
    cnn_model.add(tf.keras.layers.InputLayer(input_shape=(28, 28, 1)))
    cnn_model.add(tf.keras.layers.Conv2D(32, (5, 5), strides=1, padding='same'))
    cnn_model.add(tf.keras.layers.Activation('relu'))
    cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn_model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=1, padding='same'))
    cnn_model.add(tf.keras.layers.Activation('relu'))
    cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn_model.add(tf.keras.layers.Flatten())
    cnn_model.add(tf.keras.layers.Dense(1024))
    cnn_model.add(tf.keras.layers.Activation('relu'))
    cnn_model.add(tf.keras.layers.Dropout(0.5))
    cnn_model.add(tf.keras.layers.Dense(2))
    cnn_model.add(tf.keras.layers.Activation('softmax'))

    adam = tf.keras.optimizers.Adam(lr=lr)

    cnn_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return cnn_model

def cnn_150x150x1(lr=1e-4):
    cnn_model = tf.keras.models.Sequential()
    cnn_model.add(tf.keras.layers.InputLayer(input_shape=(150, 150, 1)))
    cnn_model.add(tf.keras.layers.Conv2D(32, (5, 5), strides=1, padding='same'))
    cnn_model.add(tf.keras.layers.Activation('relu'))
    cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn_model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=1, padding='same'))
    cnn_model.add(tf.keras.layers.Activation('relu'))
    cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn_model.add(tf.keras.layers.Flatten())
    cnn_model.add(tf.keras.layers.Dense(1024))
    cnn_model.add(tf.keras.layers.Activation('relu'))
    cnn_model.add(tf.keras.layers.Dropout(0.5))
    cnn_model.add(tf.keras.layers.Dense(2))
    cnn_model.add(tf.keras.layers.Activation('softmax'))

    adam = tf.keras.optimizers.Adam(lr=lr)

    cnn_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return cnn_model

def cnn_150x150x5(lr=1e-4):
    cnn_model = tf.keras.models.Sequential()
    cnn_model.add(tf.keras.layers.InputLayer(input_shape=(150, 150, 5)))
    cnn_model.add(tf.keras.layers.Conv2D(32, (5, 5), strides=1, padding='same'))
    cnn_model.add(tf.keras.layers.Activation('relu'))
    cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn_model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=1, padding='same'))
    cnn_model.add(tf.keras.layers.Activation('relu'))
    cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn_model.add(tf.keras.layers.Flatten())
    cnn_model.add(tf.keras.layers.Dense(1024))
    cnn_model.add(tf.keras.layers.Activation('relu'))
    cnn_model.add(tf.keras.layers.Dropout(0.5))
    cnn_model.add(tf.keras.layers.Dense(2))
    cnn_model.add(tf.keras.layers.Activation('softmax'))

    adam = tf.keras.optimizers.Adam(lr=lr)

    cnn_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return cnn_model

def cnn_150x150x5_3class(lr=1e-4):
    cnn_model = tf.keras.models.Sequential()
    cnn_model.add(tf.keras.layers.InputLayer(input_shape=(150, 150, 5)))
    cnn_model.add(tf.keras.layers.Conv2D(32, (5, 5), strides=1, padding='same'))
    cnn_model.add(tf.keras.layers.Activation('relu'))
    cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn_model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=1, padding='same'))
    cnn_model.add(tf.keras.layers.Activation('relu'))
    cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn_model.add(tf.keras.layers.Flatten())
    cnn_model.add(tf.keras.layers.Dense(1024))
    cnn_model.add(tf.keras.layers.Activation('relu'))
    cnn_model.add(tf.keras.layers.Dropout(0.5))
    cnn_model.add(tf.keras.layers.Dense(3))
    cnn_model.add(tf.keras.layers.Activation('softmax'))

    adam = tf.keras.optimizers.Adam(lr=lr)

    cnn_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return cnn_model