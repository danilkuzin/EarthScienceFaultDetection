import tensorflow as tf
import numpy as np


class  ImageAugmentation:
    @staticmethod
    def augment(x: np.array) -> np.array:
        x = tf.image.rot90(x, k=tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
        x = tf.image.random_brightness(x, 0.05)
        x = tf.image.random_contrast(x, 0.7, 1.3)

        x = ImageAugmentation.zoom(x)
        return x.numpy()

    @staticmethod
    def zoom(x: tf.Tensor) -> tf.Tensor:
        scales = list(np.arange(0.8, 1.0, 0.01))
        boxes = np.zeros((len(scales), 4))

        for i, scale in enumerate(scales):
            x1 = y1 = 0.5 - (0.5 * scale)
            x2 = y2 = 0.5 + (0.5 * scale)
            boxes[i] = [x1, y1, x2, y2]

        def random_crop(img):
            crops = tf.image.crop_and_resize([img], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(150, 150))
            return crops[tf.random_uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]

        choice = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))


