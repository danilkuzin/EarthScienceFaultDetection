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
    #todo fit generator here - use generator from DataPreprocessor.__iter__
    history = model.fit_generator(train_generator,
                        steps_per_epoch=50,
                        epochs=5,
                        validation_data=valid_generator
                        )
    # pydot not working
    # tf.keras.utils.plot_model(model, to_file='model.png')
    # model.save is not working in keras https://github.com/keras-team/keras/issues/11683
    model.save_weights('model.h5')
    # test_generator.reset()
    # pred = model.predict_generator(test_generator, verbose=1)
    #
    # predicted_class_indices = np.argmax(pred, axis=1)
    # labels = (train_generator.class_indices)
    # labels = dict((v, k) for k, v in labels.items())
    # predictions = [labels[k] for k in predicted_class_indices]
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
#    print(predictions)

def train_with_gen():
    generator_state = DataPreprocessor("../../data/Data22012019/")

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

def apply_for_sliding_window():
    data = DataPreprocessor("../../data/Region 1 - Lopukangri/", backend=Backend.GDAL, filename_prefix="tibet", mode=Mode.TEST)
    model = cnn_for_mnist_adjust_lr_with_softmax()
    model.load_weights('model.h5')

    stride = 100
    max_output_size = 20
    iou_threshold = 0.5
    score_threshold = float('-inf')

    boxes = []
    scores = []
    for top_left_border_x,top_left_border_y in tqdm(itertools.product(range(0, 21 * 150, stride), range(0, 21 * 150, stride))):
        cur_patch = data.optical_rgb[top_left_border_x:top_left_border_x+150, top_left_border_y:top_left_border_y+150]
        #todo move this proc into separate func
        patch_grsc = Image.fromarray(cur_patch).convert('L')
        patch_arr = np.array(patch_grsc)
        patch_prep = np.expand_dims(patch_arr, axis=2)  # add colour
        patch_prep = np.expand_dims(patch_prep, axis=0)  # add batch
        patch_resc = patch_prep / 255

        probs = model.predict(patch_resc)
        probs_for_first__in_batch = probs[0]
        fault_prob = probs_for_first__in_batch[0]
        # todo check if y, x, y, x are correct and not are x, y, x, y
        boxes.append([top_left_border_x,top_left_border_y, top_left_border_x+150, top_left_border_y+150])
        scores.append(fault_prob)

    selected_indices = tf.image.non_max_suppression(
        boxes,
        scores,
        max_output_size = max_output_size,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        name=None
    )

    with tf.Session() as sess:
        boxes_ind = selected_indices.eval(session=sess)

        res_im = np.zeros((22 * 150, 22 * 150, 3))
        res_im[:,:] = 0, 0, 255
        for box_ind in boxes_ind:
            c_box = boxes[box_ind]
            print("box: [{}]".format(c_box))
            res_im[c_box[0]:c_box[2],
            c_box[1]: c_box[3]] = [255, 0, 0]

        res_im_im = Image.fromarray(res_im.astype(np.uint8))
        res_im_im.save('out_slding.tif')

        mask = Image.open('out_slding.tif').convert('RGBA')
        mask_np = np.array(mask)
        mask_np[:, :, 3] = 60 * np.ones((22 * 150, 22 * 150))
        mask_a = Image.fromarray(mask_np)
        orig = Image.open(data_dir + 'data.tif')
        orig_c = orig.crop((0, 0, 22 * 150, 22 * 150))
        Image.alpha_composite(orig_c, mask_a).save("out_mask_sliding.tif")



if __name__ == "__main__":
    #train()
    #apply_for_all_patches()
    #apply_for_test()
    #apply_for_muga_puruo()
    apply_for_sliding_window()
    #combine_images()
    #combine_features_images()
