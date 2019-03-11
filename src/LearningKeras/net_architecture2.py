import tensorflow as tf

class CnnModel150x150x5(tf.keras.Model):
    def __init__(self):
        super(CnnModel150x150x5, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(150, 150, 5))
        self.conv1 = tf.keras.layers.Conv2D(32, (5, 5), strides=1, padding='same')
        self.activation1 = tf.keras.layers.Activation('relu')
        self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, (5, 5), strides=1, padding='same')
        self.activation2 =tf.keras.layers.Activation('relu')
        self.maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(1024)
        self.activation3 = tf.keras.layers.Activation('relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(2)
        self.softmax = tf.keras.layers.Activation('softmax')

    def call(self, x):
        x = self.input_layer(x)
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.activation3(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.softmax(x)
        return x


