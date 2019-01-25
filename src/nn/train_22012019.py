import cv2
import itertools

import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.DataPreprocessor.preprocess_data_22012019 import DataPreprocessor22012019
from src.nn.net import cnn_for_mnist_adjust_lr_with_softmax

data_dir = "../../data/Data22012019/"

def train():
    #todo normalise images

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        data_dir + 'learn/train',
        color_mode='grayscale',
        target_size=(150, 150),
        batch_size=32,
        shuffle=True,
        class_mode='categorical')

    valid_generator = train_datagen.flow_from_directory(
        data_dir + 'learn/valid',
        color_mode='grayscale',
        target_size=(150, 150),
        batch_size=32,
        shuffle=True,
        class_mode='categorical')

    #todo replace train_gen with smth without zooming, etc! also potentially disable dropout
    test_generator = train_datagen.flow_from_directory(
        data_dir + 'learn/test',
        color_mode='grayscale',
        target_size=(28, 28),
        batch_size=1,
        class_mode=None)

    model = cnn_for_mnist_adjust_lr_with_softmax()
    model.fit_generator(train_generator,
                        steps_per_epoch=10,
                        epochs=5,
                        validation_data=valid_generator
                        )

    # model.save is not working in keras https://github.com/keras-team/keras/issues/11683
    model.save_weights('model.h5')
    # test_generator.reset()
    # pred = model.predict_generator(test_generator, verbose=1)
    #
    # predicted_class_indices = np.argmax(pred, axis=1)
    # labels = (train_generator.class_indices)
    # labels = dict((v, k) for k, v in labels.items())
    # predictions = [labels[k] for k in predicted_class_indices]

#    print(predictions)

def train_with_gen():
    generator_state = DataPreprocessor22012019("../../data/Data22012019/")

    dataset = tf.data.Dataset.from_generator(
        lambda: generator_state,
        (tf.float32, tf.float32),
        (tf.TensorShape([None, 28, 28, 1]), tf.TensorShape([None, 2]))
    )

    model = cnn_for_mnist_adjust_lr_with_softmax()
    #model.fit(dataset.map(lambda x, y: tf.image.resize_images).make_one_shot_iterator(),
    #          steps_per_epoch=None,
    #          epochs=1,
    #          verbose=1)


def apply_for_all_patches():
    model = cnn_for_mnist_adjust_lr_with_softmax()
    model.load_weights('model.h5')
    res_im = np.zeros((22*150, 22*150, 3))
    for i, j in itertools.product(range(22), range(22)):
        patch_rgb = Image.open(data_dir+'all/{}_{}.tif'.format(i,j))
        patch_grsc = patch_rgb.convert('L')
        patch_arr = np.array(patch_grsc)
        patch_prep = np.expand_dims(patch_arr, axis=2)# add colour
        patch_prep = np.expand_dims(patch_prep, axis=0)# add batch
        res = model.predict(patch_prep)
        res = res[0]
        if res[0] > res[1]:
            res_im[i * 150: (i + 1) * 150,
                        j * 150: (j+1) * 150] = [255, 0, 0]
        else:
            res_im[i * 150: (i + 1) * 150,
            j * 150: (j+1) * 150] = [0, 0, 255]
    res_im_im = Image.fromarray(res_im.astype(np.uint8))
    res_im_im.save('out.tif')
        #print(res)

def combine_images():
    mask = Image.open('out.tif').convert('RGBA')
    mask_np = np.array(mask)
    mask_np[:,:,3] = 60*np.ones((22*150, 22*150))
    mask_a = Image.fromarray(mask_np)
    orig = Image.open(data_dir+'/data.tif')
    orig_c = orig.crop((0,0,22*150, 22*150))
    Image.alpha_composite(orig_c, mask_a).save("out_mask.tif")

def combine_features_images():
    mask = Image.open(data_dir+'/feature_categories.tif').convert('RGBA').crop((0,0,22*150, 22*150))
    mask_np = np.array(mask)
    for i1, i2 in tqdm(itertools.product(range(22*150), range(22*150))):
        if np.any(mask_np[i1, i2] == 1):
            mask_np[i1, i2] = [250, 0, 0, 0]
        if np.any(mask_np[i1, i2] == 2):
            mask_np[i1, i2] = [0, 250, 0, 0]
        if np.any(mask_np[i1, i2] ==3):
            mask_np[i1, i2] = [0, 0, 250, 0]
    mask_np[:,:,3] = 60*np.ones((22*150, 22*150))
    mask_a = Image.fromarray(mask_np)
    orig = Image.open(data_dir+'/data.tif')
    orig_c = orig.crop((0,0,22*150, 22*150))
    Image.alpha_composite(orig_c, mask_a).save("out_features_mask.tif")

if __name__ == "__main__":
    #train()
    #apply_for_all_patches()
    combine_images()
    #combine_features_images()
