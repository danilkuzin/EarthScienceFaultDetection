import itertools
import pathlib

import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from src.DataPreprocessor.PatchesOutputBackend.backend import PatchesOutputBackend
from src.DataPreprocessor.data_preprocessor import DataPreprocessor, Backend, Mode

data_dir = "../../data/Region 1 - Lopukangri/"
data_dir_muga_puruo = "../../data/Region 2 - Muga Puruo/"

class KerasTrainer:
    def __init__(self, model_generator, ensemble_size: int, data_preprocessor: DataPreprocessor, batch_size: int):
        self.model_generator = model_generator
        self.ensemble_size = ensemble_size
        self.data_preprocessor = data_preprocessor
        self.batch_size = batch_size

    def train(self, steps_per_epoch, epochs, train_generator):

        for i in range(self.ensemble_size):
            model = self.model_generator()
            history = model.fit_generator(train_generator(),
                                          steps_per_epoch=steps_per_epoch,
                                          epochs=epochs,
                                          validation_data=train_generator(),
                                          validation_steps=5)
            # pydot not working
            # tf.keras.utils.plot_model(model, to_file='model.png')
            # model.save is not working in keras https://github.com/keras-team/keras/issues/11683
            pathlib.Path('models').mkdir(parents=True, exist_ok=True)
            model.save_weights('models/model_{}.h5'.format(i))
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

    def apply_for_all_patches(self):
        #pathlib.Path('res').mkdir(parents=True, exist_ok=True)
        res_im_joint = np.zeros((self.ensemble_size, 22*150, 22*150, 3))
        for en in trange(self.ensemble_size):
            model = self.model_generator()
            model.load_weights('models/model_{}.h5'.format(en))
            res_im = np.zeros((22*150, 22*150, 3))
            for i, j in itertools.product(range(22), range(22)):
                left_border = i * 150
                right_border = (i + 1) * 150
                top_border = j * 150
                bottom_border = (j+1) * 150
                patch = self.data_preprocessor.concatenate_full_patch(left_border, right_border, top_border, bottom_border)
                res = model.predict(np.expand_dims(patch, axis=0))
                res = res[0]
                if res[0] > res[1]:
                    res_im[left_border: right_border, top_border: bottom_border] = [0, 0, 255]
                else:
                    res_im[left_border: right_border, top_border: bottom_border] = [255, 0, 0]
            res_im_joint[en] = res_im
        return res_im_joint
        #    res_im_im = Image.fromarray(res_im.astype(np.uint8))
        #    res_im_im.save('res/res_{}.tif'.format(en))

    def apply_for_test(self):
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

    def combine_images_im_memory(self, masks):
        images = np.zeros((self.ensemble_size, 150, 150, 4))
        for en in range(self.ensemble_size):
            mask_np = np.array(Image.fromarray((((masks[en] + 0.5) * 255).astype(np.uint8))).convert('RGBA'))
            mask_np[:, :, 3] = 60 * np.ones((22 * 150, 22 * 150))
            mask_a = Image.fromarray(mask_np)

            orig = Image.fromarray(self.data_preprocessor.original_optical_rgb)
            orig_c = orig.crop((0, 0, 22 * 150, 22 * 150))

            images[en] = np.array(Image.alpha_composite(orig_c, mask_a))
        return images

    def apply_for_sliding_window(self, data_preprocessor: DataPreprocessor, stride, max_output_size):

        #todo merge ensemble results
        with tf.Session() as sess:
            res = []
            for en in trange(self.ensemble_size):
                model = self.model_generator()
                model.load_weights('models/model_{}.h5'.format(en))

                iou_threshold = 0.5
                score_threshold = float('-inf')

                boxes = []
                scores = []
                for top_left_border_x, top_left_border_y in tqdm(
                        itertools.product(range(0, 21 * 150, stride), range(0, 21 * 150, stride))):
                    cur_patch = data_preprocessor.concatenate_full_patch(
                                top_left_border_x, top_left_border_x + 150,
                                top_left_border_y, top_left_border_y + 150)

                    probs = model.predict(np.expand_dims(cur_patch, axis=0))
                    probs_for_first__in_batch = probs[0]
                    fault_prob = probs_for_first__in_batch[1]
                    # todo check if y, x, y, x are correct and not are x, y, x, y
                    boxes.append([top_left_border_x, top_left_border_y, top_left_border_x + 150, top_left_border_y + 150])
                    scores.append(fault_prob)

                selected_indices = tf.image.non_max_suppression(
                    boxes,
                    scores,
                    max_output_size=max_output_size,
                    iou_threshold=iou_threshold,
                    score_threshold=score_threshold,
                    name=None
                )

                boxes_ind = selected_indices.eval(session=sess)

                res_im = np.zeros((22 * 150, 22 * 150, 3))
                res_im[:, :] = 0, 0, 255
                for box_ind in boxes_ind:
                    c_box = boxes[box_ind]
                    print("box: [{}]".format(c_box))
                    res_im[c_box[0]:c_box[2],
                    c_box[1]: c_box[3]] = [255, 0, 0]

                res_im_im = Image.fromarray(res_im.astype(np.uint8))
                #res_im_im.save('out_slding.tif')

                mask = res_im_im.convert('RGBA')
                mask_np = np.array(mask)
                mask_np[:, :, 3] = 60 * np.ones((22 * 150, 22 * 150))
                mask_a = Image.fromarray(mask_np)
                orig = Image.fromarray(data_preprocessor.original_optical_rgb).convert('RGBA')
                orig_c = orig.crop((0, 0, 22 * 150, 22 * 150))
                res.append(Image.alpha_composite(orig_c, mask_a))
            return res

def combine_images():
    mask = Image.open('out.tif').convert('RGBA')
    mask_np = np.array(mask)
    mask_np[:,:,3] = 60*np.ones((22*150, 22*150))
    mask_a = Image.fromarray(mask_np)
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
    #train()
    #apply_for_all_patches()
    #apply_for_test()
    #apply_for_muga_puruo()
    apply_for_sliding_window()
    #combine_images()
    #combine_features_images()
