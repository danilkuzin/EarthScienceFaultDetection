import tensorflow as tf

#todo change loss to something "stable"  from logits
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

def cnn_150x150x12(lr=1e-4):
    cnn_model = tf.keras.models.Sequential()
    cnn_model.add(tf.keras.layers.InputLayer(input_shape=(150, 150, 12)))
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


def cnn_150x150x11(lr=1e-4):
    cnn_model = tf.keras.models.Sequential()
    cnn_model.add(tf.keras.layers.InputLayer(input_shape=(150, 150, 11)))
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

def cnn_150x150x10(lr=1e-4):
    cnn_model = tf.keras.models.Sequential()
    cnn_model.add(tf.keras.layers.InputLayer(input_shape=(150, 150, 10)))
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

def cnn_150x150x4(lr=1e-4):
    cnn_model = tf.keras.models.Sequential()
    cnn_model.add(tf.keras.layers.InputLayer(input_shape=(150, 150, 4)))
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

def cnn_150x150x3(lr=1e-4):
    cnn_model = tf.keras.models.Sequential()
    cnn_model.add(tf.keras.layers.InputLayer(input_shape=(150, 150, 3)))
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

def alexnet(lr=1e-4):
    #from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/alexnet.py and
    # https: // github.com / pytorch / vision / blob / master / torchvision / models / alexnet.py
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(224, 224, 3)))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=[11, 11], strides=4, padding='valid'))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=[3, 3], strides=2))
    model.add(tf.keras.layers.Conv2D(filters=192, kernel_size=[5, 5]))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=[3, 3], strides=2))
    model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=[3, 3]))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3]))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3]))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=[3, 3], strides=2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=4096))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(units=4096))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2))
    model.add(tf.keras.layers.Activation('softmax'))

    # model.add(tf.keras.layers.Conv2D(filters=4096,  kernel_size=[5, 5], padding='valid'))
    # model.add(tf.keras.layers.ReLU())
    # model.add(tf.keras.layers.Dropout(rate=0.5))
    # model.add(tf.keras.layers.Conv2D(filters=4096, kernel_size=[1, 1]))
    # model.add(tf.keras.layers.ReLU())
    # model.add(tf.keras.layers.Dropout(rate=0.5))
    # model.add(tf.keras.layers.Conv2D(filters=2, kernel_size=[1, 1]))

    adam = tf.keras.optimizers.Adam(lr=lr)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def cnn_150x150x5_2class_3convolutions(lr=1e-4):
    cnn_model = tf.keras.models.Sequential()
    cnn_model.add(tf.keras.layers.InputLayer(input_shape=(150, 150, 5)))
    cnn_model.add(tf.keras.layers.Conv2D(32, (5, 5), strides=1, padding='same'))
    cnn_model.add(tf.keras.layers.Activation('relu'))
    cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn_model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=1, padding='same'))
    cnn_model.add(tf.keras.layers.Activation('relu'))
    cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn_model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=1, padding='same'))
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

#todo consider 3d convs?
def cnn_150x150x5_3class_5convolutions(lr=1e-4):
    cnn_model = tf.keras.models.Sequential()
    cnn_model.add(tf.keras.layers.InputLayer(input_shape=(150, 150, 5)))
    cnn_model.add(tf.keras.layers.Conv2D(32, (5, 5), strides=1, padding='same'))
    cnn_model.add(tf.keras.layers.Activation('relu'))
    cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn_model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=1, padding='same'))
    cnn_model.add(tf.keras.layers.Activation('relu'))
    cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn_model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=1, padding='same'))
    cnn_model.add(tf.keras.layers.Activation('relu'))
    cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn_model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=1, padding='same'))
    cnn_model.add(tf.keras.layers.Activation('relu'))
    cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    cnn_model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=1, padding='same'))
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

def cnn_150x150x3_3class(lr=1e-4):
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

def cnn_150x150x1_3class(lr=1e-4):
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