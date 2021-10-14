import torch
import torch.nn as nn
from torch.autograd import Function


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means, sds):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means)/sds

class Model(nn.Module):
    def __init__(self, normalizer=None):
        super(Model, self).__init__()

        self.normalize_layer = NormalizeLayer(list(normalizer.mean), list(normalizer.std))

        self.h_dim_1 = 64
        self.h_dim = 128
        self.output_dim = 256

        # Classifier
        self.classifier = nn.Sequential()
        self.classifier.add_module('g_conv1', nn.Conv2d(3, self.h_dim_1, kernel_size=5))
        self.classifier.add_module('g_bn1', nn.BatchNorm2d(self.h_dim_1))
        self.classifier.add_module('g_pool1', nn.MaxPool2d(2))
        self.classifier.add_module('g_relu1', nn.ReLU(True))
        self.classifier.add_module('g_conv2', nn.Conv2d(self.h_dim_1, self.h_dim, kernel_size=5))
        self.classifier.add_module('g_bn2', nn.BatchNorm2d(self.h_dim))
        self.classifier.add_module('g_pool2', nn.MaxPool2d(2))
        self.classifier.add_module('g_relu2', nn.ReLU(True))
        self.classifier.add_module('flatten', nn.Flatten())
        self.classifier.add_module('c_fc1', nn.Linear(self.h_dim * 4 * 4, self.output_dim))
        self.classifier.add_module('c_bn1', nn.BatchNorm1d(self.output_dim))
        self.classifier.add_module('c_relu1', nn.ReLU(True))
        self.classifier.add_module('c_fc2', nn.Linear(self.output_dim, 10))

        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
    
    def get_logit_output(self, input_data):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        input_data = self.normalize_layer(input_data)
        
        class_output = self.classifier(input_data)
        return class_output

    def forward(self, input_data):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        input_data = self.normalize_layer(input_data)

        class_output = self.classifier(input_data)

        return self.logsoftmax(class_output), self.softmax(class_output), None

