import numpy as np
import tensorflow as tf

from src.DataPreprocessor.PatchesOutputBackend.sampled_backend import SampledBackend


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class TfrecordBackend(SampledBackend):
    def save(self, array: np.array, label:int, path: str):
        with tf.python_io.TFRecordWriter(path + '.tfrecord') as writer:
            for index in range(arr.shape[0]):
                image_raw = arr[index].tostring()
                # todo consider recording rgb elevation separate from slope as ints to reduce data size and conversion overhead
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'height': _int64_feature(array.shape[1]),
                            'width': _int64_feature(array.shape[2]),
                            'depth': _int64_feature(array.shape[3]),
                            'label': _int64_feature(label.value),
                            # 'image_raw': _bytes_feature(image_raw)
                            'image_raw': tf.train.Feature(
                                float_list=tf.train.FloatList(value=arr[index].flatten().astype(np.float32)))
                        }))
                writer.write(example.SerializeToString())