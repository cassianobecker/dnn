import torch
import torch.nn as nn
import torch.nn.functional as func

from nn.DtiConv3d import DwiConv3dUnitKernel
from util.architecture import Dimensions


class DnnHcpUnitKernel(nn.Module):

    def __init__(self, img_dims, number_of_classes, cholesky_weights=False):
        super(DnnHcpUnitKernel, self).__init__()

        img_channels = img_dims[0]
        kernel_dims = [4, 4, 4]
        strides = [1, 1, 1]
        c_out1 = 2*4*10
        c_out2 = 3*4*10
        pool_size1 = 5
        pool_size = 2

        self.conv1 = DwiConv3dUnitKernel(img_channels, c_out1, cholesky_weights=cholesky_weights)
        self.conv2 = nn.Conv3d(c_out1, c_out2, kernel_dims, strides)
        self.conv3 = nn.Conv3d(c_out2, c_out2, kernel_dims, strides)

        self.max1 = nn.MaxPool3d(pool_size1)
        self.max2 = nn.MaxPool3d(pool_size)
        self.max3 = nn.MaxPool3d(pool_size)

        self.dropout1 = nn.Dropout3d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        self.dropout3 = nn.Dropout2d(0.25)

        linear_size1 = Dimensions().dimensions_for_linear(
            img_dims,
            [
                self.conv1, self.max1,
                self.conv2, self.max2,
                self.conv3, self.max3
             ])

        linear_size2 = 128

        self.fc1 = nn.Linear(int(linear_size1), linear_size2)
        self.fc2 = nn.Linear(linear_size2, number_of_classes)

    def scores(self, x):

        x = self.conv1(x)
        x = func.relu(x)
        x = self.max1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = func.relu(x)
        x = self.max2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = func.relu(x)
        x = self.max3(x)
        x = self.dropout3(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = func.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x

    def forward(self, x):

        y = self.scores(x)
        output = func.log_softmax(y, dim=1)

        return output
