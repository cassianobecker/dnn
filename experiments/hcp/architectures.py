import torch
import torch.nn as nn
import torch.nn.functional as func

from nn.DtiConv3d import DwiConv3dUnitKernel
from util.architecture import Dimensions


class DnnHcp(nn.Module):

    def __init__(self, img_dims, number_of_classes, cholesky_weights=False):
        super(DnnHcp, self).__init__()

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


class CnnHcp(nn.Module):

    def __init__(self, img_dims, number_of_classes, cholesky_weights=False):
        super(CnnHcp, self).__init__()

        k = 5
        s = 2

        img_channels = img_dims[0]
        kernel_dims1 = [k, k, k]
        kernel_dims2 = [k, k, k]
        strides1 = [s, s, s]
        strides2 = [s, s, s]
        c_out1 = 128
        c_out2 = 64
        pool_size = 2

        self.conv0 = nn.Conv3d(img_channels, c_out1, kernel_dims1, strides1)
        self.conv1 = nn.Conv3d(c_out1, c_out2, kernel_dims1, strides1)
        self.conv2 = nn.Conv3d(c_out2, c_out2, kernel_dims2, strides2)

        self.dropout1 = nn.Dropout3d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.max1 = nn.MaxPool3d(pool_size)

        linear_size1 = Dimensions().dimensions_for_linear(img_dims, [self.conv0, self.conv1, self.conv2, self.max1])

        #linear_size1 = 28224
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


class DnnHcpUnitKernel1(nn.Module):

    def __init__(self, img_dims, number_of_classes, cholesky_weights=False):
        super(DnnHcpUnitKernel1, self).__init__()

        k = 5
        s = 3

        img_channels = img_dims[0]
        kernel_dims1 = [k, k, k]
        kernel_dims2 = [k, k, k]
        strides1 = [s, s, s]
        strides2 = [s, s, s]
        c_out1 = 128
        c_out2 = 64
        pool_size = 2

        self.conv0 = DwiConv3dUnitKernel(img_channels, c_out1, cholesky_weights=cholesky_weights)
        self.conv1 = nn.Conv3d(c_out1, c_out2, kernel_dims1, strides1)
        self.conv2 = nn.Conv3d(c_out2, c_out2, kernel_dims2, strides2)

        self.dropout1 = nn.Dropout3d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.max1 = nn.MaxPool3d(pool_size)

        linear_size1 = Dimensions().dimensions_for_linear(img_dims, [self.conv0, self.conv1, self.conv2, self.max1])

        # linear_size1 = 28224
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


# larger pool size, smaller stride
class DnnHcpUnitKernel2(nn.Module):

    def __init__(self, img_dims, number_of_classes, cholesky_weights=False):
        super(DnnHcpUnitKernel2, self).__init__()

        k = 5
        s = 2

        img_channels = img_dims[0]
        kernel_dims1 = [k, k, k]
        kernel_dims2 = [k, k, k]
        strides1 = [s, s, s]
        strides2 = [s, s, s]
        c_out1 = 128
        c_out2 = 64
        pool_size = 4

        self.conv0 = DwiConv3dUnitKernel(img_channels, c_out1, cholesky_weights=cholesky_weights)
        self.conv1 = nn.Conv3d(c_out1, c_out2, kernel_dims1, strides1)
        self.conv2 = nn.Conv3d(c_out2, c_out2, kernel_dims2, strides2)

        self.dropout1 = nn.Dropout3d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.max1 = nn.MaxPool3d(pool_size)

        linear_size1 = Dimensions().dimensions_for_linear(img_dims, [self.conv0, self.conv1, self.conv2, self.max1])

        # linear_size1 = 28224
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


# larger kernel, larger pool size
class DnnHcpUnitKernel3(nn.Module):

    def __init__(self, img_dims, number_of_classes, cholesky_weights=False):
        super(DnnHcpUnitKernel3, self).__init__()

        k = 7
        s = 2

        img_channels = img_dims[0]
        kernel_dims1 = [k, k, k]
        kernel_dims2 = [k, k, k]
        strides1 = [s, s, s]
        strides2 = [s, s, s]
        c_out1 = 128
        c_out2 = 64
        pool_size = 5

        self.conv0 = DwiConv3dUnitKernel(img_channels, c_out1, cholesky_weights=cholesky_weights)
        self.conv1 = nn.Conv3d(c_out1, c_out2, kernel_dims1, strides1)
        self.conv2 = nn.Conv3d(c_out2, c_out2, kernel_dims2, strides2)

        self.dropout1 = nn.Dropout3d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.max1 = nn.MaxPool3d(pool_size)

        linear_size1 = Dimensions().dimensions_for_linear(img_dims, [self.conv0, self.conv1, self.conv2, self.max1])

        # linear_size1 = 28224
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


# less dropout
class DnnHcpUnitKernel4(nn.Module):

    def __init__(self, img_dims, number_of_classes, cholesky_weights=False):
        super(DnnHcpUnitKernel4, self).__init__()

        k = 5
        s = 3

        img_channels = img_dims[0]
        kernel_dims1 = [k, k, k]
        kernel_dims2 = [k, k, k]
        strides1 = [s, s, s]
        strides2 = [s, s, s]
        c_out1 = 128
        c_out2 = 64
        pool_size = 2

        self.conv0 = DwiConv3dUnitKernel(img_channels, c_out1, cholesky_weights=cholesky_weights)
        self.conv1 = nn.Conv3d(c_out1, c_out2, kernel_dims1, strides1)
        self.conv2 = nn.Conv3d(c_out2, c_out2, kernel_dims2, strides2)

        self.dropout1 = nn.Dropout3d(0.2)
        self.dropout2 = nn.Dropout2d(0.2)
        self.max1 = nn.MaxPool3d(pool_size)

        linear_size1 = Dimensions().dimensions_for_linear(img_dims, [self.conv0, self.conv1, self.conv2, self.max1])

        # linear_size1 = 28224
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


# less dropout
class DnnHcpUnitKernelShallow0(nn.Module):

    def __init__(self, img_dims, number_of_classes, cholesky_weights=False):
        super(DnnHcpUnitKernelShallow0, self).__init__()

        k = 5
        s = 3

        img_channels = img_dims[0]
        kernel_dims1 = [k, k, k]
        kernel_dims2 = [k, k, k]
        strides1 = [s, s, s]
        strides2 = [s, s, s]
        c_out1 = 128
        c_out2 = 64
        pool_size = 4

        self.conv0 = DwiConv3dUnitKernel(img_channels, c_out1, cholesky_weights=cholesky_weights)
        self.conv1 = nn.Conv3d(c_out1, c_out2, kernel_dims1, strides1)
        # self.conv2 = nn.Conv3d(c_out2, c_out2, kernel_dims2, strides2)

        self.dropout1 = nn.Dropout3d(0.2)
        self.dropout2 = nn.Dropout2d(0.2)
        self.max1 = nn.MaxPool3d(pool_size)

        linear_size1 = Dimensions().dimensions_for_linear(img_dims, [self.conv0, self.conv1, self.max1])

        # linear_size1 = 28224
        linear_size2 = 128

        self.fc1 = nn.Linear(int(linear_size1), linear_size2)
        self.fc2 = nn.Linear(linear_size2, number_of_classes)

    def forward(self, x):

        x = self.conv0(x)

        x = self.conv1(x)
        x = func.relu(x)

        # x = self.conv2(x)
        # x = func.relu(x)
        x = self.max1(x)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = func.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        output = func.log_softmax(x, dim=1)

        return output


# less dropout
class DnnHcpUnitKernelShallow1(nn.Module):

    def __init__(self, img_dims, number_of_classes, cholesky_weights=False):
        super(DnnHcpUnitKernelShallow1, self).__init__()

        k = 5
        s = 3

        img_channels = img_dims[0]
        kernel_dims1 = [k, k, k]
        kernel_dims2 = [k, k, k]
        strides1 = [s, s, s]
        strides2 = [s, s, s]
        c_out1 = 2*128
        c_out2 = 2*64
        pool_size = 4

        self.conv0 = DwiConv3dUnitKernel(img_channels, c_out1, cholesky_weights=cholesky_weights)
        self.conv1 = nn.Conv3d(c_out1, c_out2, kernel_dims1, strides1)
        # self.conv2 = nn.Conv3d(c_out2, c_out2, kernel_dims2, strides2)

        self.dropout1 = nn.Dropout3d(0.2)
        self.dropout2 = nn.Dropout2d(0.2)
        self.max1 = nn.MaxPool3d(pool_size)

        linear_size1 = Dimensions().dimensions_for_linear(img_dims, [self.conv0, self.conv1, self.max1])

        # linear_size1 = 28224
        linear_size2 = 128

        self.fc1 = nn.Linear(int(linear_size1), linear_size2)
        self.fc2 = nn.Linear(linear_size2, number_of_classes)

    def forward(self, x):

        x = self.conv0(x)

        x = self.conv1(x)
        x = func.relu(x)

        # x = self.conv2(x)
        # x = func.relu(x)
        x = self.max1(x)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = func.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        output = func.log_softmax(x, dim=1)

        return output


# less dropout
class DnnHcpUnitKernelShallow2(nn.Module):

    def __init__(self, img_dims, number_of_classes, cholesky_weights=False):
        super(DnnHcpUnitKernelShallow2, self).__init__()

        k = 5
        s = 3

        img_channels = img_dims[0]
        kernel_dims1 = [k, k, k]
        kernel_dims2 = [k, k, k]
        strides1 = [s, s, s]
        strides2 = [s, s, s]
        c_out1 = 128
        c_out2 = 64
        pool_size = 2

        self.conv0 = DwiConv3dUnitKernel(img_channels, c_out1, cholesky_weights=cholesky_weights)
        self.conv1 = nn.Conv3d(c_out1, c_out2, kernel_dims1, strides1)
        # self.conv2 = nn.Conv3d(c_out2, c_out2, kernel_dims2, strides2)

        self.dropout1 = nn.Dropout3d(0.2)
        self.dropout2 = nn.Dropout2d(0.2)
        self.max1 = nn.MaxPool3d(pool_size)

        linear_size1 = Dimensions().dimensions_for_linear(img_dims, [self.conv0, self.conv1, self.max1])

        # linear_size1 = 28224
        linear_size2 = 128

        self.fc1 = nn.Linear(int(linear_size1), linear_size2)
        self.fc2 = nn.Linear(linear_size2, number_of_classes)

    def forward(self, x):

        x = self.conv0(x)
        x = func.relu(x)

        x = self.conv1(x)
        x = func.relu(x)

        # x = self.conv2(x)
        # x = func.relu(x)
        x = self.max1(x)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = func.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        output = func.log_softmax(x, dim=1)

        return output


class DnnHcpUnitKernelShallow3(nn.Module):

    def __init__(self, img_dims, number_of_classes, cholesky_weights=False):
        super(DnnHcpUnitKernelShallow3, self).__init__()

        k = 5
        s = 3

        img_channels = img_dims[0]
        kernel_dims1 = [k, k, k]
        kernel_dims2 = [k, k, k]
        strides1 = [s, s, s]
        strides2 = [s, s, s]
        c_out1 = 2*128
        c_out2 = 2*64
        pool_size = 2

        self.conv0 = DwiConv3dUnitKernel(img_channels, c_out1, cholesky_weights=cholesky_weights)
        self.conv1 = nn.Conv3d(c_out1, c_out2, kernel_dims1, strides1)
        # self.conv2 = nn.Conv3d(c_out2, c_out2, kernel_dims2, strides2)

        self.dropout1 = nn.Dropout3d(0.2)
        self.dropout2 = nn.Dropout2d(0.2)
        self.max1 = nn.MaxPool3d(pool_size)

        linear_size1 = Dimensions().dimensions_for_linear(img_dims, [self.conv0, self.conv1, self.max1])

        # linear_size1 = 28224
        linear_size2 = 128

        self.fc1 = nn.Linear(int(linear_size1), linear_size2)
        self.fc2 = nn.Linear(linear_size2, number_of_classes)

    def forward(self, x):

        x = self.conv0(x)
        x = func.relu(x)

        x = self.conv1(x)
        x = func.relu(x)

        # x = self.conv2(x)
        # x = func.relu(x)
        x = self.max1(x)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = func.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        output = func.log_softmax(x, dim=1)

        return output

# big dropout
class DnnHcpUnitKernelShallow4(nn.Module):

    def __init__(self, img_dims, number_of_classes, cholesky_weights=False):
        super(DnnHcpUnitKernelShallow4, self).__init__()

        k = 5
        s = 3

        img_channels = img_dims[0]
        kernel_dims1 = [k, k, k]
        kernel_dims2 = [k, k, k]
        strides1 = [s, s, s]
        strides2 = [s, s, s]
        c_out1 = 2*128
        c_out2 = 2*64
        pool_size = 2

        self.conv0 = DwiConv3dUnitKernel(img_channels, c_out1, cholesky_weights=cholesky_weights)
        self.conv1 = nn.Conv3d(c_out1, c_out2, kernel_dims1, strides1)
        # self.conv2 = nn.Conv3d(c_out2, c_out2, kernel_dims2, strides2)

        self.dropout1 = nn.Dropout3d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.max1 = nn.MaxPool3d(pool_size)

        linear_size1 = Dimensions().dimensions_for_linear(img_dims, [self.conv0, self.conv1, self.max1])

        # linear_size1 = 28224
        linear_size2 = 128

        self.fc1 = nn.Linear(int(linear_size1), linear_size2)
        self.fc2 = nn.Linear(linear_size2, number_of_classes)

    def forward(self, x):

        x = self.conv0(x)

        x = self.conv1(x)
        x = func.relu(x)

        # x = self.conv2(x)
        # x = func.relu(x)
        x = self.max1(x)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = func.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        output = func.log_softmax(x, dim=1)

        return output

# no maxpool
class DnnHcpUnitKernelShallow5(nn.Module):

    def __init__(self, img_dims, number_of_classes, cholesky_weights=False):
        super(DnnHcpUnitKernelShallow5, self).__init__()

        k = 5
        s = 3

        img_channels = img_dims[0]
        kernel_dims1 = [k, k, k]
        kernel_dims2 = [k, k, k]
        strides1 = [s, s, s]
        strides2 = [s, s, s]
        c_out1 = 2*128
        c_out2 = 2*64
        # pool_size = 1

        self.conv0 = DwiConv3dUnitKernel(img_channels, c_out1, cholesky_weights=cholesky_weights)
        self.conv1 = nn.Conv3d(c_out1, c_out2, kernel_dims1, strides1)
        # self.conv2 = nn.Conv3d(c_out2, c_out2, kernel_dims2, strides2)

        self.dropout1 = nn.Dropout3d(0.2)
        self.dropout2 = nn.Dropout2d(0.2)
        # self.max1 = nn.MaxPool3d(pool_size)

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

        # x = self.conv2(x)
        # x = func.relu(x)
        x = self.max1(x)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = func.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        output = func.log_softmax(x, dim=1)

        return output


# low stride
class DnnHcpUnitKernelShallow6(nn.Module):

    def __init__(self, img_dims, number_of_classes, cholesky_weights=False):
        super(DnnHcpUnitKernelShallow6, self).__init__()

        k = 5
        s = 1

        img_channels = img_dims[0]
        kernel_dims1 = [k, k, k]
        kernel_dims2 = [k, k, k]
        strides1 = [s, s, s]
        strides2 = [s, s, s]
        c_out1 = 2*128
        c_out2 = 2*64
        pool_size = 3

        self.conv0 = DwiConv3dUnitKernel(img_channels, c_out1, cholesky_weights=cholesky_weights)
        self.conv1 = nn.Conv3d(c_out1, c_out2, kernel_dims1, strides1)
        # self.conv2 = nn.Conv3d(c_out2, c_out2, kernel_dims2, strides2)

        self.dropout1 = nn.Dropout3d(0.3)
        self.dropout2 = nn.Dropout2d(0.3)
        self.max1 = nn.MaxPool3d(pool_size)

        linear_size1 = Dimensions().dimensions_for_linear(img_dims, [self.conv0, self.conv1, self.max1])

        # linear_size1 = 28224
        linear_size2 = 128

        self.fc1 = nn.Linear(int(linear_size1), linear_size2)
        self.fc2 = nn.Linear(linear_size2, number_of_classes)

    def forward(self, x):

        x = self.conv0(x)
        # x = func.relu(x)

        x = self.conv1(x)
        x = func.relu(x)

        # x = self.conv2(x)
        # x = func.relu(x)
        x = self.max1(x)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = func.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        output = func.log_softmax(x, dim=1)

        return output