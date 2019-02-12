import itertools
import pathlib
from typing import List

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm, trange

from src.DataPreprocessor.data_preprocessor import DataPreprocessor, FeatureValue


class KerasTrainer:
    def __init__(self, model_generator, ensemble_size: int):
        self.model_generator = model_generator
        self.ensemble_size = ensemble_size
        self.models = []

    def train(self, steps_per_epoch, epochs, train_generator):
        history_arr = []
        for i in range(self.ensemble_size):
            model = self.model_generator()
            history = model.fit_generator(train_generator,
                                          steps_per_epoch=steps_per_epoch,
                                          epochs=epochs,
                                          validation_data=train_generator,
                                          validation_steps=5,
                                          workers=0,
                                          use_multiprocessing=False)

            self.models.append(model)
            history_arr.append(history)
        return history_arr

    def save(self, output_path):
        for i in range(self.ensemble_size):
            pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
            self.models[i].save_weights(output_path + '/model_{}.h5'.format(i))

    def load(self, input_path):
        for i in range(self.ensemble_size):
            model = self.model_generator()
            model.load_weights('{}/model_{}.h5'.format(input_path, i))
            self.models.append(model)

    def predict_average(self, patch):
        res_arr = []
        for model in self.models:
            res_arr.append(model.predict(patch))
        res_np = np.concatenate(res_arr)
        res_avg = np.mean(res_np, axis=0)
        return res_avg

    def apply_for_test(self, data_generator, num_test_samples):
        true_labels = []
        predicted_labels = []

        for _ in trange(num_test_samples):
            images, lbls = next(data_generator)
            for i in range(images.shape[0]):
                probs = self.predict_average(images[i])
                true_labels.append(lbls[i])
                predicted_labels.append(np.argmax(probs))

        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)

        return np.mean(true_labels == predicted_labels)

    def apply_for_sliding_window(self, data_preprocessor: DataPreprocessor, patch_size: List[int, int], ):
        #todo in progress, refactoring of other methods goes here
        boxes, probs = [], []
        #todo maybe create a getter in data_preprocessor that returns shape of the full image, or better iterator that
        # moves for the image with stridwes and returns sequentially, like generator but not random
        for patch_coords_batch, patch_batch in data_preprocessor.sequential_pass_generator:
            #returns a batch of images!
            boxes.append(patch_coords_batch)
            probs.append(self.predict_average(patch_batch))
        probs = np.array(probs)
        avg_probs =

        for top_left_border_x, top_left_border_y in itertools.product(range(0, 21 * 150, stride), range(0, 21 * 150, stride)):
            boxes.append([top_left_border_x, top_left_border_y, top_left_border_x + 150, top_left_border_y + 150])

    def apply_for_sliding_window_3class_batch(self, data_preprocessor: DataPreprocessor, stride, batch_size):
        boxes, avg_fault_probs, avg_lookalike_probs, avg_non_fault_probs = [], [], [], []
        for top_left_border_x, top_left_border_y in itertools.product(range(0, 21 * 150, stride), range(0, 21 * 150, stride)):
            boxes.append([top_left_border_x, top_left_border_y, top_left_border_x + 150, top_left_border_y + 150])

        images_batch = []
        for borders in tqdm(boxes):
            top_left_x, top_left_y, bottom_right_x, bottom_right_y = borders
            cur_patch = data_preprocessor.concatenate_full_patch(top_left_x, bottom_right_x, top_left_y, bottom_right_y)
            images_batch.append(cur_patch)

        images_batch = np.array(images_batch)

        probs = np.zeros((self.ensemble_size, images_batch.shape[0], 3))
        for (ind, model) in enumerate(self.models):
            for batch_ind in trange(int(images_batch.shape[0] / batch_size)):
                probs_batch = model.predict(images_batch[batch_ind*batch_size:(batch_ind+1)*batch_size])
                probs[ind, batch_ind*batch_size:(batch_ind+1)*batch_size] = probs_batch

        #probs = np.array(probs)
        avg_probs = np.mean(probs, axis=0)
        avg_fault_probs = avg_probs[:, FeatureValue.FAULT.value].tolist()
        avg_lookalike_probs = avg_probs[:, FeatureValue.FAULT_LOOKALIKE.value].tolist()
        avg_non_fault_probs = avg_probs[:, FeatureValue.NONFAULT.value].tolist()

        return boxes, avg_fault_probs, avg_lookalike_probs, avg_non_fault_probs

    def apply_for_sliding_window_2class_batch(self, data_preprocessor: DataPreprocessor, stride, batch_size):
        boxes, avg_fault_probs, avg_non_fault_probs = [], [], []
        for top_left_border_x, top_left_border_y in itertools.product(range(0, 21 * 150, stride), range(0, 21 * 150, stride)):
            boxes.append([top_left_border_x, top_left_border_y, top_left_border_x + 150, top_left_border_y + 150])

        images_batch = []
        for borders in tqdm(boxes):
            top_left_x, top_left_y, bottom_right_x, bottom_right_y = borders
            cur_patch = data_preprocessor.concatenate_full_patch(top_left_x, bottom_right_x, top_left_y, bottom_right_y)
            images_batch.append(cur_patch)

        images_batch = np.array(images_batch)

        probs = np.zeros((self.ensemble_size, images_batch.shape[0], 2))
        for (ind, model) in enumerate(self.models):
            for batch_ind in trange(int(images_batch.shape[0] / batch_size)):
                probs_batch = model.predict(images_batch[batch_ind*batch_size:(batch_ind+1)*batch_size])
                probs[ind, batch_ind*batch_size:(batch_ind+1)*batch_size] = probs_batch

        #probs = np.array(probs)
        avg_probs = np.mean(probs, axis=0)
        avg_fault_probs = avg_probs[:, FeatureValue.FAULT.value].tolist()
        avg_non_fault_probs = avg_probs[:, FeatureValue.FAULT_LOOKALIKE.value].tolist() # thats because the class numbers were 0 and 1...

        return boxes, avg_fault_probs, avg_non_fault_probs

    def apply_for_sliding_window_non_max_suppression(self, boxes, avg_fault_probs, max_output_size, data_preprocessor: DataPreprocessor):
        # todo check if y, x, y, x are correct and not are x, y, x, y
        iou_threshold = 0.5
        score_threshold = float('-inf')

        scores = avg_fault_probs

        with tf.Session() as sess:
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
                res_im[c_box[0]:c_box[2],
                c_box[1]: c_box[3]] = [255, 0, 0]

            res_im_im = Image.fromarray(res_im.astype(np.uint8))
            #res_im_im.save('out_slding.tif')

            mask = res_im_im.convert('RGBA')
            mask_np = np.array(mask)
            mask_np[:, :, 3] = 60 * np.ones((22 * 150, 22 * 150))
            mask_a = Image.fromarray(mask_np)
            orig = Image.fromarray(data_preprocessor.optical_rgb).convert('RGBA')
            orig_c = orig.crop((0, 0, 22 * 150, 22 * 150))
        return Image.alpha_composite(orig_c, mask_a)

    # todo for each pixel try with normalised product of probabilities for each class, not max
    # todo check parameters, like patch size and their amount, process them from function inputs
    def apply_for_sliding_window_heatmaps(self, boxes, avg_fault_probs, data_preprocessor: DataPreprocessor):
        res_im = np.zeros((22 * 150, 22 * 150), dtype=np.float)
        for (index, borders) in enumerate(tqdm(boxes)):
            top_left_x, top_left_y, bottom_right_x, bottom_right_y = borders
            res_im[top_left_x:bottom_right_x, top_left_y:bottom_right_y] = np.maximum(res_im[top_left_x:bottom_right_x, top_left_y:bottom_right_y],
                                                                                      avg_fault_probs[index] * np.ones_like(res_im[top_left_x:bottom_right_x, top_left_y:bottom_right_y]))

            #for i,j in itertools.product(range(150), range(150)):
            #    res_im[top_left_x+i, top_left_y+j] = np.max((res_im[top_left_x+i, top_left_y+j], avg_fault_probs[index]))

        # for (index, borders) in enumerate(tqdm(boxes)):
        #     top_left_x, top_left_y, bottom_right_x, bottom_right_y = borders
        #     for i,j in itertools.product(range(150), range(150)):
        #         res_im[top_left_x+i, top_left_y+j] = np.max((res_im[top_left_x+i, top_left_y+j], avg_fault_probs[index]))
        # res_im_im = Image.fromarray(res_im.astype(np.uint8))
        # mask = res_im_im.convert('RGBA')
        # mask_np = np.array(mask)
        # mask_np[:, :, 3] = 60 * np.ones((22 * 150, 22 * 150))
        # mask_a = Image.fromarray(mask_np)
        # orig = Image.fromarray(data_preprocessor.optical_rgb).convert('RGBA')
        # orig_c = orig.crop((0, 0, 22 * 150, 22 * 150))
        # return Image.alpha_composite(orig_c, mask_a)
        return res_im



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
