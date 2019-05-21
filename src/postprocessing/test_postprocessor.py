from src.postprocessing.postprocessor import PostProcessor
import numpy as np

def generate_4x_boxes():
    original_2dimage_shape = (4, 4)
    stride = (1, 1)
    patch_shape = (2, 2)
    boxes_in_row = int((original_2dimage_shape[0] - patch_shape[0]) / stride[0]) + 1
    boxes_in_column = int((original_2dimage_shape[1] - patch_shape[1]) / stride[1]) + 1
    n_boxes = boxes_in_row * boxes_in_column
    boxes = np.zeros((n_boxes, 4), dtype=np.int)

    for i in range(boxes_in_row):
        for j in range(boxes_in_column):
            top_left_x, top_left_y, bottom_right_x, bottom_right_y = i*stride[0], j*stride[1], i*stride[0]+patch_shape[0], j*stride[1] + patch_shape[1]
            boxes[i*boxes_in_row + j] = np.array([top_left_x, top_left_y, bottom_right_x, bottom_right_y], dtype=np.int)

    return boxes, original_2dimage_shape

def generate_4x_probs():
    pixel_probs = np.array([[0.5, 1, 0, 0.5],
                      [1, 1, 0.5, 0.5],
                      [0, 0, 1, 1],
                      [0.5, 0.5, 1, 1]])
    box_probs = np.zeros((9,))
    for i in range(3):
        for j in range(3):
            box_probs[i*3+ j] = pixel_probs[i] * pixel_probs
    true_im =

def test_heatmaps():
    boxes, original_2dimage_shape = generate_4x_boxes()
    probs, true_im = generate_4x_probs()
    postprocessor = PostProcessor(boxes=boxes, original_2dimage_shape=original_2dimage_shape, probs=probs)
    im = postprocessor.heatmaps(mode="mean")
    np.testing.assert_allclose(actual=im, desired=true_im)