import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.DtiConv3d import DtiConv3dTorch


class DNN1(nn.Module):

    def __init__(self):
        super(DNN1, self).__init__()

        #img_dims = [28, 28, lift_dim]
        img_dims = [145, 174, 145]

        kernel_dims1 = [10, 10, 10]
        kernel_dims2 = [5, 5, 5]

        c_out1 = 2*4*10
        c_out2 = 3*4*10
        skip = 1

        self.pool_width = 2

        linear_size1 = (c_out2 / (self.pool_width ** 3)) *\
                       (img_dims[0] - kernel_dims1[0] - 1) *\
                       (img_dims[1] - kernel_dims1[1] - 1) *\
                       (img_dims[2] - kernel_dims1[2])

        linear_size1 = 513000
        # linear_size1 = 217800
        linear_size2 = 128
        number_of_classes = 2
        strides = [4, 4, 4]

        self.conv1 = DtiConv3dTorch(c_out1, kernel_dims1, strides)

        self.conv2 = nn.Conv3d(c_out1, c_out2, kernel_dims2, skip)

        self.conv3 = nn.Conv3d(c_out2, c_out2, kernel_dims2, skip)

        self.dropout1 = nn.Dropout3d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        # self.dropout3 = nn.Dropout2d(0.25)

        self.fc1 = nn.Linear(int(linear_size1), linear_size2)
        self.fc2 = nn.Linear(linear_size2, number_of_classes)

    def forward(self, x):

        x = x * 3000
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool3d(x, self.pool_width)
        x = self.dropout1(x)
        # x = F.relu(x)
        # x = self.conv3(x)
        # x = self.dropout3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)

        return output


class DNN2(nn.Module):

    def __init__(self):
        super(DNN2, self).__init__()

        #img_dims = [28, 28, lift_dim]
        img_dims = [145, 174, 145]

        kernel_dims1 = [10, 10, 10]
        kernel_dims2 = [5, 5, 5]

        c_out1 = 2*4*10
        c_out2 = 3*4*10
        skip = 1

        self.pool_width = 2

        linear_size1 = (c_out2 / (self.pool_width ** 3)) *\
                       (img_dims[0] - kernel_dims1[0] - 1) *\
                       (img_dims[1] - kernel_dims1[1] - 1) *\
                       (img_dims[2] - kernel_dims1[2])

        linear_size1 = 513000
        linear_size2 = 128
        number_of_classes = 2
        strides = [4, 4, 4]

        self.conv1 = DtiConv3dTorch(c_out1, kernel_dims1, strides)

        self.conv2 = nn.Conv3d(c_out1, c_out2, kernel_dims2, skip)

        self.conv3 = nn.Conv3d(c_out2, c_out2, kernel_dims2, skip)

        self.dropout1 = nn.Dropout3d(0.2)
        self.dropout2 = nn.Dropout2d(0.1)

        self.fc1 = nn.Linear(int(linear_size1), linear_size2)
        self.fc2 = nn.Linear(linear_size2, number_of_classes)

    def forward(self, x):

        x = x * 3000
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool3d(x, self.pool_width)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)

        return output