import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.optim import lr_scheduler


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def init_layer(layer, weight_init=None, bias_init=None):
    if weight_init is not None:
        layer.weight.data = torch.FloatTensor(weight_init)
    if bias_init is not None:
        layer.bias.data = torch.FloatTensor(bias_init)
    return layer


def nn_Linear(in_features, out_features, bias=True, weight_init=None,
              bias_init=None):
    layer = nn.Linear(in_features, out_features, bias)
    return init_layer(layer, weight_init, bias_init)


def nn_Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
              dilation=1, groups=1,
              bias=True, weight_init=None, bias_init=None):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      dilation,
                      groups, bias)
    return init_layer(layer, weight_init, bias_init)


def cnn_150x150x5_torch(lr=1e-4):
    cnn_model = nn.Sequential()
    #cnn_model.add_module(tf.keras.layers.InputLayer(input_shape=(150, 150, 5)))
    cnn_model.add_module('conv1', nn_Conv2d(
        in_channels=5,
        out_channels=32,
        kernel_size=(5, 5),
        stride=1,
        padding=4,
        dilation=1,
        bias=False,
        weight_init=None))
    cnn_model.add_module('relu1', nn.ReLU())
    cnn_model.add_module('mp1', nn.MaxPool2d(
        kernel_size=2,
        stride=2,
        padding=0
    ))
    cnn_model.add_module('conv2', nn_Conv2d(
        in_channels=32,
        out_channels=64,
        kernel_size=(5, 5),
        stride=1,
        padding=4))
    cnn_model.add_module('relu2', nn.ReLU())
    cnn_model.add_module('mp2', nn.MaxPool2d(
        kernel_size=2,
        stride=2,
        padding=0
    ))
    cnn_model.add_module('flatten', Flatten())
    cnn_model.add_module('fc1', nn_Linear(
        in_features=64*40*40,
        out_features=1024,
        bias=False,
        weight_init=None
    ))
    cnn_model.add_module('fc1_relu', nn.ReLU())
    cnn_model.add_module('fc1_dp', nn.Dropout(0.5))
    cnn_model.add_module('fc2', nn_Linear(
        in_features=1024,
        out_features=1,
        bias=True,
        weight_init=None,
        bias_init=None
    ))
    cnn_model.add_module('probs', nn.Sigmoid())

    criterion = nn.BCELoss()
    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer = optim.Adam(cnn_model.parameters(), lr=1e-4)

    # Decay LR by a factor of 0.1 every 10 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10,
                                           gamma=0.1)

    return cnn_model, criterion, optimizer, exp_lr_scheduler


def cnn_150x150x5_fully_conv_torch(path, lr=1e-4):
    # path - path to flatten net weights
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    flatten_model_output = torch.load(path, map_location=device)
    flatten_model = flatten_model_output['model']
    cnn_model = nn.Sequential()
    #cnn_model.add_module(tf.keras.layers.InputLayer(input_shape=(150, 150, 5)))
    cnn_model.add_module('conv1', nn_Conv2d(
        in_channels=5,
        out_channels=32,
        kernel_size=(5, 5),
        stride=1,
        padding=2,
        dilation=1,
        bias=False,
        weight_init=flatten_model.conv1.weight.data))
    cnn_model.add_module('relu1', nn.ReLU())
    cnn_model.add_module('pad1', nn.ConstantPad2d((0, 1, 0, 1), 0))
    cnn_model.add_module('mp1', nn.MaxPool2d(
        kernel_size=2,
        stride=1,
        padding=0
    ))
    cnn_model.add_module('conv2', nn_Conv2d(
        in_channels=32,
        out_channels=64,
        kernel_size=(5, 5),
        stride=1,
        padding=4,
        dilation=2,
        bias=False,
        weight_init=flatten_model.conv2.weight.data))
    cnn_model.add_module('relu2', nn.ReLU())
    cnn_model.add_module('pad2', nn.ConstantPad2d((0, 2, 0, 2), 0))
    cnn_model.add_module('mp2', nn.MaxPool2d(
        kernel_size=2,
        stride=1,
        padding=0
    ))
    cnn_model.add_module('pad3', nn.ConstantPad2d(
        (4*19, 4*20-1, 4*19, 4*20-1), 0))
    cnn_model.add_module('fc1', nn_Conv2d(
        in_channels=64, #*40*40,
        out_channels=1024,
        kernel_size=(40, 40),
        stride=1,
        padding=0, #4*19,
        dilation=4,
        bias=False,
        weight_init=torch.reshape(flatten_model.fc1.weight.data,
                                  (1024, 64, 40, 40))
    ))
    cnn_model.add_module('fc1_relu', nn.ReLU())
    cnn_model.add_module('fc1_dp', nn.Dropout(0.5))
    cnn_model.add_module('fc2', nn_Conv2d(
        in_channels=1024,
        out_channels=1,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        weight_init=torch.reshape(flatten_model.fc2.weight.data,
                                  (1, 1024, 1, 1)),
        bias_init=flatten_model.fc2.bias.data
    ))
    cnn_model.add_module('probs', nn.Sigmoid())

    cnn_model = cnn_model.train(False)

    # criterion = nn.BCELoss()
    # # Observe that only parameters of final layer are being optimized as
    # # opposed to before.
    # optimizer = optim.Adam(cnn_model.parameters(), lr=1e-4)
    #
    # # Decay LR by a factor of 0.1 every 10 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10,
    #                                        gamma=0.1)

    return cnn_model #, criterion, optimizer, exp_lr_scheduler


class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=5,
            out_channels=32,
            kernel_size=(5, 5),
            bias=False,
            padding=2)
        self.relu1 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            return_indices=True
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(5, 5),
            bias=False,
            padding=2
        )
        self.relu2 = nn.ReLU()
        # self.pad2 = nn.ConstantPad2d((0, 1, 0, 1), 0)
        self.mp2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            return_indices=True
        )
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(35, 35),
            bias=False,
        )
        self.relu3 = nn.ReLU()
        self.conv3transp = nn.ConvTranspose2d(
            out_channels=64,
            in_channels=128,
            kernel_size=(35, 35),
            bias=False
        )
        self.relu4 = nn.ReLU()
        self.m2up = nn.MaxUnpool2d(
            kernel_size=2
        )
        # self.unpad2 = nn.ConstantPad2d((0, -1, 0, -1), 0)
        self.conv2transp = nn.ConvTranspose2d(
            out_channels=32,
            in_channels=64,
            kernel_size=(5, 5),
            bias=False,
            padding=2
        )
        self.relu5 = nn.ReLU()
        self.m1up = nn.MaxUnpool2d(
            kernel_size=2
        )
        self.conv1transp = nn.ConvTranspose2d(
            out_channels=1,
            in_channels=32,
            kernel_size=(5, 5),
            bias=False,
            padding=2
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        if (x.shape[2] % 2) == 0:
            pad1_first = 0
        else:
            pad1_first = 1
        if (x.shape[3] % 2) == 0:
            pad1_second = 0
        else:
            pad1_second = 1
        x = nn.ConstantPad2d((0, pad1_first, 0, pad1_second), 0)(x)
        x, mp1_indices = self.mp1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        if (x.shape[2] % 2) == 0:
            pad2_first = 0
        else:
            pad2_first = 1
        if (x.shape[3] % 2) == 0:
            pad2_second = 0
        else:
            pad2_second = 1
        x = nn.ConstantPad2d((0, pad2_first, 0, pad2_second), 0)(x)
        x, mp2_indices = self.mp2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv3transp(x)
        x = self.relu4(x)
        x = self.m2up(x, mp2_indices)
        x = nn.ConstantPad2d((0, -pad2_first, 0, -pad2_second), 0)(x)
        # x = self.unpad2(x)
        x = self.conv2transp(x)
        x = self.relu5(x)
        x = self.m1up(x, mp1_indices)
        x = nn.ConstantPad2d((0, -pad1_first, 0, -pad1_second), 0)(x)
        x = self.conv1transp(x)
        x = torch.squeeze(x)
        return x

# def cnn_150x150x5_fully_conv_with_transposes_torch(input):
#     # cnn_model = nn.Sequential()
#     #cnn_model.add_module(tf.keras.layers.InputLayer(input_shape=(150, 150, 5)))
#     conv1 = nn_Conv2d(
#         in_channels=5,
#         out_channels=32,
#         kernel_size=(5, 5),
#         bias=False)
#     relu1 = cnn_model.add_module('relu1', nn.ReLU())
#     cnn_model.add_module('mp1', nn.MaxPool2d(
#         kernel_size=2,
#         return_indices=True
#     ))
#     cnn_model.add_module('conv2', nn_Conv2d(
#         in_channels=32,
#         out_channels=64,
#         kernel_size=(5, 5),
#         bias=False))
#     cnn_model.add_module('relu2', nn.ReLU())
#     cnn_model.add_module('mp2', nn.MaxPool2d(
#         kernel_size=2,
#         return_indices=True
#     ))
#     cnn_model.add_module('conv3', nn_Conv2d(
#         in_channels=64,
#         out_channels=128,
#         kernel_size=(34, 34),
#         bias=False
#     ))
#     cnn_model.add_module('relu3', nn.ReLU())
#     cnn_model.add_module('conv3transp', nn.ConvTranspose2d(
#         out_channels=64,
#         in_channels=128,
#         kernel_size=(34, 34),
#         bias=False
#     ))
    # cnn_model.add_module('relu4', nn.ReLU())
    # cnn_model.add_module('m2up', nn.MaxUnpool2d(
    #     kernel_size=2
    # ))
    # cnn_model.add_module('conv2transp', nn.ConvTranspose2d(
    #     out_channels=32,
    #     in_channels=64,
    #     kernel_size=(5, 5),
    #     bias=False
    # ))
    # cnn_model.add_module('relu5', nn.ReLU())
    # cnn_model.add_module('m1up', nn.MaxUnpool2d(
    #     kernel_size=2
    # ))
    # cnn_model.add_module('conv1transp', nn.ConvTranspose2d(
    #     out_channels=2,
    #     in_channels=32,
    #     kernel_size=(5, 5),
    #     bias=False
    # ))
    #
    # cnn_model.add_module('probs', nn.Softmax2d())

    # cnn_model = cnn_model.train(False)

    # criterion = nn.CrossEntropyLoss()
    # # Observe that only parameters of final layer are being optimized as
    # # opposed to before.
    # optimizer = optim.Adam(cnn_model.parameters(), lr=1e-4)
    #
    # # Decay LR by a factor of 0.1 every 10 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10,
    #                                        gamma=0.1)
    #
    # return cnn_model, criterion, optimizer, exp_lr_scheduler

