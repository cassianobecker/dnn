import torch
import torch.nn as nn
import torch.nn.functional as func

from nn.DtiConv3d import DwiConv3dUnitKernel
from util.architecture import Dimensions


class DnnMnistUnitKernel(nn.Module):

    def __init__(self, img_dims, number_of_classes, cholesky_weights=False):
        super(DnnMnistUnitKernel, self).__init__()

        img_channels = img_dims[0]
        kernel_dims1 = [3, 3, 3]
        kernel_dims2 = [3, 3, 2]
        strides1 = [1, 1, 1]
        strides2 = [1, 1, 1]
        c_out1 = 32
        c_out2 = 64
        pool_size = 2

        self.conv0 = DwiConv3dUnitKernel(img_channels, c_out1, cholesky_weights=cholesky_weights)
        self.conv1 = nn.Conv3d(c_out1, c_out2, kernel_dims1, strides1)
        self.conv2 = nn.Conv3d(c_out2, c_out2, kernel_dims2, strides2)

        self.dropout1 = nn.Dropout3d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.max1 = nn.MaxPool3d(pool_size)

        # linear_size1 = Dimensions().dimensions_for_linear(img_dims, [self.conv0, self.conv1, self.conv2, self.max1])

        linear_size1 = 18432
        linear_size2 = 128

        self.fc1 = nn.Linear(int(linear_size1), linear_size2)
        self.fc2 = nn.Linear(linear_size2, number_of_classes)

    def forward(self, x):

        x = self.conv0(x)

        x = self.conv1(x)
        x = func.relu(x)

        x = self.conv2(x)
        x = func.relu(x)
        x = self.max1(x)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = func.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        output = func.log_softmax(x, dim=1)

        return output


class DnnMnist(nn.Module):

    def __init__(self, img_dims, number_of_classes, cholesky_weights=False):
        super(DnnMnist, self).__init__()

        img_channels = img_dims[0]
        kernel_dims1 = [3, 3, 3]
        kernel_dims2 = [3, 3, 2]
        strides1 = [1, 1, 1]
        strides2 = [1, 1, 1]
        c_out1 = 32
        c_out2 = 64
        pool_size = 2

        self.conv0 = nn.Conv3d(img_channels, c_out1,  kernel_dims1, strides1)
        self.conv1 = nn.Conv3d(c_out1, c_out2, kernel_dims1, strides1)
        self.conv2 = nn.Conv3d(c_out2, c_out2, kernel_dims2, strides2)

        self.dropout1 = nn.Dropout3d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.max1 = nn.MaxPool3d(pool_size)

        # linear_size1 = Dimensions().dimensions_for_linear(img_dims, [self.conv0, self.conv1, self.conv2, self.max1])

        linear_size1 = 7744
        linear_size2 = 128

        self.fc1 = nn.Linear(int(linear_size1), linear_size2)
        self.fc2 = nn.Linear(linear_size2, number_of_classes)

    def forward(self, x):

        x = self.conv0(x)
        x = func.relu(x)

        x = self.conv1(x)
        x = func.relu(x)

        x = self.conv2(x)
        x = func.relu(x)
        x = self.max1(x)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = func.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        output = func.log_softmax(x, dim=1)

        return output