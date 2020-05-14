import torch
import torch.nn as nn
import torch.nn.functional as func

from nn.DtiConv3d import DwiConv3dTorch, DwiConv3dTorchVect, DwiConv3dTorchVectFirst, DwiConv3dTorchVectChol
from util.architecture import Dimensions


class DNN1(nn.Module):

    def __init__(self, img_dims, number_of_classes, half_precision=False):
        super(DNN1, self).__init__()

        img_channels = img_dims[0]
        kernel_dims1 = [6, 6, 6]
        kernel_dims2 = [5, 5, 5]
        strides1 = [3, 3, 3]
        strides2 = [2, 2, 2]
        c_out1 = 2*4*10
        c_out2 = 3*4*10
        pool_size = 2

        self.conv1 = DwiConv3dTorch(img_channels, c_out1, kernel_dims1, strides1, half_precision=half_precision)
        self.conv2 = nn.Conv3d(c_out1, c_out2, kernel_dims2, strides2)
        # self.conv3 = nn.Conv3d(c_out2, c_out2, kernel_dims2, strides2)
        self.max1 = nn.MaxPool3d(pool_size)
        self.dropout1 = nn.Dropout3d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        # self.dropout3 = nn.Dropout2d(0.25)
        # output_dims = Dimensions().dims_for_layers(img_dims, [self.conv1, self.conv2, self.max1])
        linear_size1 = Dimensions().dimensions_for_linear(img_dims[1:], [self.conv1, self.conv2, self.max1])
        linear_size2 = 128

        self.fc1 = nn.Linear(int(linear_size1), linear_size2)
        self.fc2 = nn.Linear(linear_size2, number_of_classes)

    def forward(self, x):

        # x = x * 3000
        x = self.conv1(x)
        x = func.relu(x)
        x = self.conv2(x)
        x = self.max1(x)
        x = self.dropout1(x)
        # x = F.relu(x)
        # x = self.conv3(x)
        # x = self.dropout3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = func.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        output = func.log_softmax(x, dim=1)

        return output


class DNN2(nn.Module):

    def __init__(self, img_dims, number_of_classes, half_precision=False):
        super(DNN2, self).__init__()

        img_channels = img_dims[0]
        kernel_dims1 = [3, 3, 2]
        kernel_dims2 = [2, 2, 2]
        strides1 = [2, 2, 2]
        strides2 = [2, 2, 2]
        c_out1 = 2*4*10
        c_out2 = 3*4*10
        pool_size = 2

        # self.conv1 = DwiConv3dTorchVectChol(img_channels, c_out1, half_precision=half_precision)
        self.conv1 = DwiConv3dTorchVect(img_channels, c_out1, kernel_dims1, strides1, half_precision=half_precision)
        self.conv2 = nn.Conv3d(c_out1, c_out2, kernel_dims2, strides2)
        # self.conv3 = nn.Conv3d(c_out2, c_out2, kernel_dims2, strides2)
        self.max1 = nn.MaxPool3d(pool_size)
        self.dropout1 = nn.Dropout3d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        # self.dropout3 = nn.Dropout2d(0.25)
        linear_size1 = Dimensions().dimensions_for_linear(img_dims, [self.conv1, self.conv2, self.max1])
        linear_size2 = 128

        self.fc1 = nn.Linear(int(linear_size1), linear_size2)
        self.fc2 = nn.Linear(linear_size2, number_of_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = func.relu(x)
        x = self.conv2(x)
        x = self.max1(x)
        x = self.dropout1(x)
        # x = func.relu(x)
        # x = self.conv3(x)
        # x = self.dropout3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = func.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        output = func.log_softmax(x, dim=1)

        return output


class DNN2Conv1(nn.Module):

    def __init__(self, img_dims, number_of_classes, half_precision=False):
        super(DNN2Conv1, self).__init__()

        img_channels = img_dims[0]
        kernel_dims1 = [1, 1, 1]
        kernel_dims2 = [5, 5, 5]
        strides1 = [1, 1, 1]
        strides2 = [5, 5, 5]
        c_out1 = 2*4*10
        c_out2 = 3*4*10
        pool_size = 5

        self.conv1 = DwiConv3dTorchVectFirst(img_channels, c_out1, half_precision=half_precision)
        self.conv2 = nn.Conv3d(c_out1, c_out2, kernel_dims2, strides2)
        # self.conv3 = nn.Conv3d(c_out2, c_out2, kernel_dims2, strides2)
        self.max1 = nn.MaxPool3d(pool_size)
        self.max2 = nn.MaxPool3d(pool_size)
        self.dropout1 = nn.Dropout3d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.dropout3 = nn.Dropout2d(0.25)
        linear_size1 = Dimensions().dimensions_for_linear(img_dims, [self.conv1,  self.max1, self.conv2, self.max2])
        linear_size2 = 128

        self.fc1 = nn.Linear(int(linear_size1), linear_size2)
        self.fc2 = nn.Linear(linear_size2, number_of_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = func.relu(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.dropout1(x)
        x = func.relu(x)
        x = self.max2(x)
        # x = self.conv3(x)
        # x = self.dropout3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = func.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        output = func.log_softmax(x, dim=1)

        return output


class DNN2ConvDeep1(nn.Module):

    def __init__(self, img_dims, number_of_classes, half_precision=False):
        super(DNN2ConvDeep1, self).__init__()

        img_channels = img_dims[0]
        kernel_dims = [4, 4, 4]
        strides = [1, 1, 1]
        c_out1 = 2*4*10
        c_out2 = 3*4*10
        pool_size1 = 5
        pool_size = 2

        self.conv1 = DwiConv3dTorchVectFirst(img_channels, c_out1, half_precision=half_precision)
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

    def forward(self, x):

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

        output = func.log_softmax(x, dim=1)

        return output


class DNN3(nn.Module):

    def __init__(self, img_dims, number_of_classes, half_precision=False):
        super(DNN3, self).__init__()

        img_channels = img_dims[0]
        kernel_dims1 = [6, 6, 6]
        kernel_dims2 = [5, 5, 5]
        strides1 = [3, 3, 3]
        strides2 = [2, 2, 2]
        c_out1 = 2*4*10
        c_out2 = 3*4*10
        pool_size = 2

        self.conv1 = DwiConv3dTorchVect(img_channels, c_out1, kernel_dims1, strides1, half_precision=half_precision)
        self.conv2 = nn.Conv3d(c_out1, c_out2, kernel_dims2, strides2)
        self.conv3 = nn.Conv3d(c_out2, c_out2, kernel_dims2, strides2)
        self.max1 = nn.MaxPool3d(pool_size)
        self.dropout1 = nn.Dropout3d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.dropout3 = nn.Dropout2d(0.25)
        linear_size1 = Dimensions().dimensions_for_linear(img_dims, [self.conv1, self.conv2, self.max1, self.conv3])
        linear_size2 = 128

        self.fc1 = nn.Linear(int(linear_size1), linear_size2)
        self.fc2 = nn.Linear(linear_size2, number_of_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = func.relu(x)
        x = self.conv2(x)
        x = self.max1(x)
        x = self.dropout1(x)
        x = func.relu(x)
        x = self.conv3(x)
        x = self.dropout3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = func.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        output = func.log_softmax(x, dim=1)

        return output