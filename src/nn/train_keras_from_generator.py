import cv2
import itertools

import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from src.DataPreprocessor.data_preprocessor import DataPreprocessor, Backend, Mode
from src.nn.net import cnn_for_mnist_adjust_lr_with_softmax

data_dir = "../../data/Region 1 - Lopukangri/"
data_dir_muga_puruo = "../../data/Region 2 - Muga Puruo/"

#todo add ensembling
def train():
    #todo normalise images

    model = cnn_for_mnist_adjust_lr_with_softmax()

    loader = DataPreprocessor("../../data/Region 1 - Lopukangri/", backend=Backend.GDAL, filename_prefix="tibet",
                              mode=Mode.TRAIN)

    def train_generator():
        batch_size = 10
        # todo at least, make grayscale and not red only
        while True:
            img_batch = np.zeros((batch_size, 150, 150, 1))
            lbl_batch = np.zeros((batch_size, 2))
            for i in range(batch_size):
                class_label = np.random.binomial(1, p=0.5, size=1)
                if class_label == 1:
                    patch = loader.sample_fault_patch()
                    img_batch[i] = np.expand_dims(patch[:, :, 0] / 255, axis = 2)
                    lbl_batch[i] = np.array([1, 0])
                else:
                    patch = loader.sample_nonfault_patch()
                    img_batch[i] = np.expand_dims(patch[:, :, 0] / 255, axis = 2)
                    lbl_batch[i] = np.array([0, 1])
            yield img_batch, lbl_batch

    history = model.fit_generator(train_generator(), steps_per_epoch=10,  epochs=50, validation_data=None)
    # pydot not working
    # tf.keras.utils.plot_model(model, to_file='model.png')
    # model.save is not working in keras https://github.com/keras-team/keras/issues/11683
    model.save_weights('model_from_gen.h5')
    # test_generator.reset()
    # pred = model.predict_generator(test_generator, verbose=1)
    #
    # predicted_class_indices = np.argmax(pred, axis=1)
    # labels = (train_generator.class_indices)
    # labels = dict((v, k) for k, v in labels.items())
    # predictions = [labels[k] for k in predicted_class_indices]
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    #plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
#    print(predictions)

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
        patch_resc = patch_prep / 255
        res = model.predict(patch_resc)
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

def apply_for_test():
    model = cnn_for_mnist_adjust_lr_with_softmax()
    model.load_weights('model.h5')
    res_faults = np.zeros((1000, 2))
    res_nonfaults = np.zeros((1000, 2))

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

    for i in trange(1000):
        patch_rgb = Image.open(data_dir+'learn/test_with_labels/fault/{}.tif'.format(i))
        patch_grsc = patch_rgb.convert('L')
        patch_arr = np.array(patch_grsc)
        patch_prep = np.expand_dims(patch_arr, axis=2)# add colour
        patch_prep = np.expand_dims(patch_prep, axis=0)# add batch
        patch_resc = patch_prep / 255
        res_faults[i] = model.predict(patch_resc)

    for i in trange(1000):
        patch_rgb = Image.open(data_dir + 'learn/test_with_labels/nonfault/{}.tif'.format(i))
        patch_grsc = patch_rgb.convert('L')
        patch_arr = np.array(patch_grsc)
        patch_prep = np.expand_dims(patch_arr, axis=2)  # add colour
        patch_prep = np.expand_dims(patch_prep, axis=0)  # add batch
        patch_resc = patch_prep / 255
        res_nonfaults[i] = model.predict(patch_resc)

    labels = (train_generator.class_indices)
    labels = dict((v, k) for k, v in labels.items())

    predicted_class_indices = np.argmax(res_faults, axis=1)
    predictions_faults = np.array([labels[k] for k in predicted_class_indices])
    predicted_class_indices = np.argmax(res_nonfaults, axis=1)
    predictions_nonfaults = np.array([labels[k] for k in predicted_class_indices])

    print(np.mean(predictions_faults == 'fault'))
    print(np.mean(predictions_nonfaults == 'nonfault'))


def combine_images():
    mask = Image.open('out.tif').convert('RGBA')
    mask_np = np.array(mask)
    mask_np[:,:,3] = 60*np.ones((22*150, 22*150))
    mask_a = Image.fromarray(mask_np)
    orig = Image.open(data_dir+'data.tif')
    orig_c = orig.crop((0,0,22*150, 22*150))
    Image.alpha_composite(orig_c, mask_a).save("out_mask.tif")

def apply_for_muga_puruo():
    model = cnn_for_mnist_adjust_lr_with_softmax()
    model.load_weights('model.h5')
    res_im = np.zeros((22 * 150, 22 * 150, 3))
    for i, j in tqdm(itertools.product(range(22), range(22))):
        patch_rgb = Image.open(data_dir_muga_puruo + 'all/{}_{}.tif'.format(i, j))
        patch_grsc = patch_rgb.convert('L')
        patch_arr = np.array(patch_grsc)
        patch_prep = np.expand_dims(patch_arr, axis=2)  # add colour
        patch_prep = np.expand_dims(patch_prep, axis=0)  # add batch
        patch_resc = patch_prep / 255
        res = model.predict(patch_resc)
        res = res[0]
        if res[0] > res[1]:
            res_im[i * 150: (i + 1) * 150,
            j * 150: (j + 1) * 150] = [255, 0, 0]
        else:
            res_im[i * 150: (i + 1) * 150,
            j * 150: (j + 1) * 150] = [0, 0, 255]
    res_im_im = Image.fromarray(res_im.astype(np.uint8))
    res_im_im.save('out_muga_puruo.tif')

    mask = Image.open('out_muga_puruo.tif').convert('RGBA')
    mask_np = np.array(mask)
    mask_np[:, :, 3] = 60 * np.ones((22 * 150, 22 * 150))
    mask_a = Image.fromarray(mask_np)
    orig = Image.open(data_dir_muga_puruo + 'data.tif')
    orig_c = orig.crop((0, 0, 22 * 150, 22 * 150))
    Image.alpha_composite(orig_c, mask_a).save("out_mask_muga_puruo.tif")


if __name__ == "__main__":
    train()
    #apply_for_all_patches()
    #apply_for_test()
    #apply_for_muga_puruo()
    #combine_images()
    #combine_features_images()
