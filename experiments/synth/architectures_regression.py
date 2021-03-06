import torch
import torch.nn as nn
import torch.nn.functional as func

from nn.DtiConv3d import DwiConv3dUnitKernel
from util.architecture import Dimensions


class DnnHcpUnitKernelRegression(nn.Module):

    def __init__(self, img_dims, number_of_classes, cholesky_weights=False):
        super(DnnHcpUnitKernelRegression, self).__init__()

        k = 5
        s = 2

        img_channels = img_dims[0]
        kernel_dims1 = [k, k, k]
        kernel_dims2 = [k, k, k]
        strides1 = [s, s, s]
        strides2 = [s, s, s]
        c_out1 = 128
        c_out2 = 64
        pool_size = 3

        self.conv0 = DwiConv3dUnitKernel(img_channels, c_out1, cholesky_weights=cholesky_weights)
        self.conv1 = nn.Conv3d(c_out1, c_out2, kernel_dims1, strides1)
        # self.conv2 = nn.Conv3d(c_out2, c_out2, kernel_dims2, strides2)

        # self.dropout1 = nn.Dropout3d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        # self.max1 = nn.MaxPool3d(pool_size)

        # linear_size1 = Dimensions().dimensions_for_linear(img_dims, [self.conv0, self.conv1, self.max1, self.conv2])
        linear_size1 = Dimensions().dimensions_for_linear(img_dims, [self.conv0, self.conv1])

        # linear_size1 = 28224
        linear_size2 = 128

        self.fc1 = nn.Linear(int(linear_size1), linear_size2)
        self.fc2 = nn.Linear(linear_size2, number_of_classes)

    def forward(self, x):

        x = self.conv0(x)
        # x = func.relu(x)

        x = self.conv1(x)
        x = func.relu(x)
        # x = self.max1(x)

        # x = self.conv2(x)
        # x = func.relu(x)

        # x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = func.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        # output = func.log_softmax(x, dim=1)

        output = x[:, 0]

        return output


class DnnHcpUnitKernelRegression2(nn.Module):

    def __init__(self, img_dims, number_of_classes, cholesky_weights=False):
        super(DnnHcpUnitKernelRegression2, self).__init__()

        k = 4
        s = 1

        img_channels = img_dims[0]
        kernel_dims1 = [k, k, k]
        kernel_dims2 = [k, k, k]
        strides1 = [s, s, s]
        strides2 = [s, s, s]
        c_out1 = 64
        c_out2 = 32
        pool_size = 2

        self.conv0 = DwiConv3dUnitKernel(img_channels, c_out1, cholesky_weights=cholesky_weights)
        self.conv1 = nn.Conv3d(c_out1, c_out2, kernel_dims1, strides1)
        self.conv2 = nn.Conv3d(c_out2, c_out2, kernel_dims2, strides2)

        self.dropout1 = nn.Dropout3d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.max1 = nn.MaxPool3d(pool_size)
        self.max2 = nn.MaxPool3d(pool_size)

        linear_size1 = Dimensions().dimensions_for_linear(img_dims, [self.conv0,  self.max1, self.conv1, self.max1, self.conv2, self.max2])
        # linear_size1 = Dimensions().dimensions_for_linear(img_dims, [self.conv0, self.conv1])

        # linear_size1 = 28224
        linear_size2 = 2*128

        self.fc1 = nn.Linear(int(linear_size1), linear_size2)
        self.fc2 = nn.Linear(linear_size2, number_of_classes)

    def forward(self, x):

        x = self.conv0(x)
        # x = func.relu(x)
        x = self.max1(x)

        x = self.conv1(x)
        x = func.relu(x)
        x = self.max1(x)

        x = self.conv2(x)
        x = func.relu(x)

        x = self.max2(x)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = func.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        # output = func.log_softmax(x, dim=1)

        output = x[:, 0]

        return output


class DnnSynthRegression(nn.Module):

    def __init__(self, img_dims, number_of_classes, cholesky_weights=False):
        super(DnnSynthRegression, self).__init__()

        k = 4
        s = 1

        img_channels = img_dims[0]
        kernel_dims1 = [k, k, k]
        kernel_dims2 = [k, k, k]
        strides1 = [s, s, s]
        strides2 = [s, s, s]
        c_out1 = 64
        c_out2 = 32
        pool_size = 2

        self.conv0 = nn.Conv3d(img_channels, c_out1, kernel_dims1, strides1)
        self.conv1 = nn.Conv3d(c_out1, c_out2, kernel_dims1, strides1)
        self.conv2 = nn.Conv3d(c_out2, c_out2, kernel_dims2, strides2)

        self.dropout1 = nn.Dropout3d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.max1 = nn.MaxPool3d(pool_size)
        self.max2 = nn.MaxPool3d(pool_size)

        linear_size1 = Dimensions().dimensions_for_linear(img_dims, [self.conv0,  self.max1, self.conv1, self.max1, self.conv2, self.max2])
        # linear_size1 = Dimensions().dimensions_for_linear(img_dims, [self.conv0, self.conv1])

        # linear_size1 = 28224
        linear_size2 = 2*128

        self.fc1 = nn.Linear(int(linear_size1), linear_size2)
        self.fc2 = nn.Linear(linear_size2, number_of_classes)

    def forward(self, x):

        x = self.conv0(x)
        # x = func.relu(x)
        x = self.max1(x)

        x = self.conv1(x)
        x = func.relu(x)
        x = self.max1(x)

        x = self.conv2(x)
        x = func.relu(x)

        x = self.max2(x)
        # x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = func.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)

        # output = func.log_softmax(x, dim=1)

        output = x[:, 0]

        return output


class DnnSynthRegressionSmall(nn.Module):

    def __init__(self, img_dims, number_of_classes, cholesky_weights=False):
        super(DnnSynthRegressionSmall, self).__init__()

        k = 3
        s = 1

        img_channels = img_dims[0]
        kernel_dims1 = [k, k, 1]
        kernel_dims2 = [k, k, k]
        strides1 = [s, s, s]
        strides2 = [s, s, s]
        c_out1 = 32
        c_out2 = 16
        pool_size = 2

        self.conv0 = nn.Conv3d(img_channels, c_out1, kernel_dims1, strides1)
        self.conv1 = nn.Conv3d(c_out1, c_out2, kernel_dims1, strides1)

        self.dropout1 = nn.Dropout3d(0.05)
        self.dropout2 = nn.Dropout2d(0.05)
        self.max1 = nn.MaxPool3d((pool_size, pool_size, 1))
        self.max2 = nn.MaxPool3d((pool_size, pool_size, 1))

        linear_size1 = Dimensions().dimensions_for_linear(img_dims, [self.conv0,  self.max1, self.conv1, self.max2])
        # linear_size1 = Dimensions().dimensions_for_linear(img_dims, [self.conv0, self.conv1])

        # linear_size1 = 28224
        linear_size2 = 2*128

        self.fc1 = nn.Linear(int(linear_size1), linear_size2)
        self.fc2 = nn.Linear(linear_size2, number_of_classes)

    def forward(self, x):

        x = self.conv0(x)
        # x = func.relu(x)
        x = self.max1(x)

        x = self.conv1(x)
        x = func.relu(x)
        # x = torch.tanh(x)
        x = self.max2(x)

        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)

        x = func.relu(x)
        # x = torch.tanh(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        # output = func.log_softmax(x, dim=1)

        output = x[:, 0]

        return output


class DnnSynthRegressionSmallMultiple(nn.Module):

    def __init__(self, img_dims, number_of_classes, number_of_outputs=1, cholesky_weights=False):
        super(DnnSynthRegressionSmallMultiple, self).__init__()

        k = 3
        s = 1

        img_channels = img_dims[0]
        kernel_dims1 = [k, k, 1]
        kernel_dims2 = [k, k, k]
        strides1 = [s, s, s]
        strides2 = [s, s, s]
        c_out1 = 32
        c_out2 = 16
        pool_size = 2

        self.conv0 = nn.Conv3d(img_channels, c_out1, kernel_dims1, strides1)
        self.conv1 = nn.Conv3d(c_out1, c_out2, kernel_dims1, strides1)

        self.dropout1 = nn.Dropout3d(0.05)
        self.dropout2 = nn.Dropout2d(0.05)
        self.max1 = nn.MaxPool3d((pool_size, pool_size, 1))
        self.max2 = nn.MaxPool3d((pool_size, pool_size, 1))

        linear_size1 = Dimensions().dimensions_for_linear(img_dims, [self.conv0,  self.max1, self.conv1, self.max2])
        # linear_size1 = Dimensions().dimensions_for_linear(img_dims, [self.conv0, self.conv1])

        # linear_size1 = 28224
        linear_size2 = 2*128

        self.fc1 = nn.Linear(int(linear_size1), linear_size2)
        # self.fc2 = nn.Linear(linear_size2, number_of_classes)
        self.fc2 = torch.nn.ModuleList([nn.Linear(linear_size2, number_of_classes) for _ in range(number_of_outputs)])

    def forward(self, x):

        x = self.conv0(x)
        # x = func.relu(x)
        x = self.max1(x)

        x = self.conv1(x)
        x = func.relu(x)
        # x = torch.tanh(x)
        x = self.max2(x)

        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)

        x = func.relu(x)
        # x = torch.tanh(x)
        x = self.dropout2(x)
        # x = self.fc2(x)

        heads = [head(x)[:, 0] for head in self.fc2]

        return torch.stack(heads)
