import pathlib

from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

from src.DataPreprocessor.region_dataset import RegionDataset, FeatureValue
from src.DataPreprocessor.region_normaliser import RegionNormaliser
from src.config import data_preprocessor_params

class NnVisualisation:
    """
    based on https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.4-visualizing-what-convnets-learn.ipynb
    """
    def __init__(self, model, region_id: int, num_samples: int):
        self.model = model
        self.region_id = region_id
        self.num_samples = num_samples

    def visualise_intermediate_activations(self, output_path, image, image_ind):
        layer_before_flatten_ind = 6

        # Extracts the outputs of the top layers:
        layer_outputs = [layer.output for layer in self.model.layers[:layer_before_flatten_ind]]
        # Creates a model that will return these outputs, given the model input:
        activation_model = models.Model(inputs=self.model.input, outputs=layer_outputs)
        # This will return a list of 5 Numpy arrays:
        # one array per layer activation
        activations = activation_model.predict(image)

        layer_names = []
        for layer in self.model.layers[:layer_before_flatten_ind]:
            layer_names.append(layer.name)

        images_per_row = 16

        # Now let's display our feature maps
        for layer_name, layer_activation in zip(layer_names, activations):
            # This is the number of features in the feature map
            n_features = layer_activation.shape[-1]

            # The feature map has shape (1, size, size, n_features)
            size = layer_activation.shape[1]

            # We will tile the activation channels in this matrix
            n_cols = n_features // images_per_row
            display_grid = np.zeros((size * n_cols, images_per_row * size))

            # We'll tile each filter into this big horizontal grid
            for col in range(n_cols):
                for row in range(images_per_row):
                    channel_image = layer_activation[0,
                                    :, :,
                                    col * images_per_row + row]
                    # Post-process the feature to make it visually palatable
                    channel_image -= channel_image.mean()
                    channel_image /= channel_image.std()
                    channel_image *= 64
                    channel_image += 128
                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                    display_grid[col * size: (col + 1) * size,
                    row * size: (row + 1) * size] = channel_image

            # Display the grid
            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            plt.colorbar()

            plt.savefig(f"{output_path}intermediate_activations_{layer_name}_image_{image_ind}.png")

    def visualise_convnet_filters(self, output_path):

        def deprocess_image(x):
            # normalize tensor: center on 0., ensure std is 0.1
            x -= x.mean()
            x /= (x.std() + 1e-5)
            x *= 0.1

            # clip to [0, 1]
            x += 0.5
            x = np.clip(x, 0, 1)

            # convert to RGB array
            x *= 255
            x = np.clip(x, 0, 255).astype('uint8')
            return x

        # def deprocess_image2(x):
        #     x[:, :, :3] = ((x[:, :, :3] + 0.5) * 255).astype(np.uint8)
        #     return x

        def generate_pattern(layer_name, filter_index, size=150):
            # Build a loss function that maximizes the activation
            # of the nth filter of the layer considered.
            layer_output = self.model.get_layer(layer_name).output
            loss = K.mean(layer_output[:, :, :, filter_index])

            # Compute the gradient of the input picture wrt this loss
            grads = K.gradients(loss, self.model.input)[0]

            # Normalization trick: we normalize the gradient
            grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

            # This function returns the loss and grads given the input picture
            iterate = K.function([self.model.input], [loss, grads])

            # We start from a gray image with some noise
            input_img_data = np.random.random((1, size, size, 5))

            # Run gradient ascent for 40 steps
            step = 0.1
            for i in range(50):
                loss_value, grads_value = iterate([input_img_data])
                input_img_data += grads_value * step

            img = input_img_data[0]
            img[:, :, :3] = deprocess_image(img[:, :, :3])
            img[:, :, 3] = deprocess_image(img[:, :, 3])
            img[:, :, 4] = deprocess_image(img[:, :, 4])
            return img

        for layer_name in ['conv2d', 'conv2d_1']:
            size = 150
            margin = 5

            # This a empty (black) image where we will store our results.
            results = np.zeros((4 * size + 3 * margin, 4 * size + 3 * margin, 5))

            for i in range(4):  # iterate over the rows of our results grid
                for j in range(4):  # iterate over the columns of our results grid
                    # Generate the pattern for filter `i + (j * 8)` in `layer_name`
                    filter_img = generate_pattern(layer_name, i + (j * 4), size=size)

                    # Put the result in the square `(i, j)` of the results grid
                    horizontal_start = i * size + i * margin
                    horizontal_end = horizontal_start + size
                    vertical_start = j * size + j * margin
                    vertical_end = vertical_start + size
                    results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

            res_rgb = results[:, :, 0:3]
            rgb_nonzero_idx = res_rgb > 0.1
            min_rgb = np.min(res_rgb[rgb_nonzero_idx])
            res_rgb = np.where(results[:, :, 0:3] < 0.1, min_rgb, results[:, :, 0:3])
            # Display the results grid
            plt.figure()
            plt.imshow(res_rgb)
            plt.savefig(f"{output_path}convnet_filters_rgb_{layer_name}.png")
            plt.figure()
            plt.imshow(results[:,:,3])
            plt.savefig(f"{output_path}convnet_filters_elevation_{layer_name}.png")
            plt.figure()
            plt.imshow(results[:,:,4])
            plt.savefig(f"{output_path}convnet_filters_slope_{layer_name}.png")

    def visualise_heatmaps_activations(self, output_path, image, image_ind):
        # This is the "fault" entry in the prediction vector
        african_elephant_output = self.model.output[:, 0]

        # The is the output feature map of the `conv2d_1` layer,
        # the last convolutional layer in VGG16
        last_conv_layer = self.model.get_layer('conv2d_1')

        # This is the gradient of the "african elephant" class with regard to
        # the output feature map of `block5_conv3`
        grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

        # This is a vector of shape (512,), where each entry
        # is the mean intensity of the gradient over a specific feature map channel
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

        # This function allows us to access the values of the quantities we just defined:
        # `pooled_grads` and the output feature map of `block5_conv3`,
        # given a sample image
        iterate = K.function([self.model.input], [pooled_grads, last_conv_layer.output[0]])

        # These are the values of these two quantities, as Numpy arrays,
        # given our sample image of two elephants
        pooled_grads_value, conv_layer_output_value = iterate([image])

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the elephant class
        for i in range(64):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

        # The channel-wise mean of the resulting feature map
        # is our heatmap of class activation
        heatmap = np.mean(conv_layer_output_value, axis=-1)

        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)

        fig = plt.figure(frameon=False, dpi=1)
        fig.set_size_inches(150, 150)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.matshow(heatmap, aspect='auto')
        plt.savefig(f"{output_path}heatmaps_activations_{image_ind}.png")
        plt.close(fig)

    def visualise_nn(self):
        output_path = "../nn_visualisations/"
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

        data_preprocessor = RegionDataset(self.region_id)
        normaliser = RegionNormaliser(region_id=self.region_id, area_ind=data_preprocessor_params[self.region_id][1])

        for i in range(self.num_samples):
            image, _ = data_preprocessor.sample_patch(label=FeatureValue.FAULT.value, patch_size=(150, 150))

            den_image = normaliser.denormalise_patch(image)
            den_rgb, den_elev, den_slope = den_image[:,:,:3], den_image[:,:,3], den_image[:,:,4]
            plt.imsave(output_path + f"input_image_{i}_0_2.png", den_rgb.astype(np.uint8))
            plt.imsave(output_path + f"input_image_{i}_3.png", den_elev)
            plt.imsave(output_path + f"input_image_{i}_4.png", den_slope)

            image = image[:, :, 0:5]
            image = np.expand_dims(image, axis=0)
            self.visualise_intermediate_activations(output_path, image, i)
            self.visualise_convnet_filters(output_path)
            self.visualise_heatmaps_activations(output_path, image, i)



